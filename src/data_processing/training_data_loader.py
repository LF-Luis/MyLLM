import os
import re
import logging
import torch
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict

from src.utils.rand_idx_seq_gen import RandIdxSeqGen


log = logging.getLogger(__name__)

SHARD_RE_PATTERN = re.compile(r"_i_(\d+)_t_(\d+)\.npy")
SHARD_RND_SEED_BASE = 42
SAMPLE_RND_SEED_BASE = 42


@dataclass
class Shard:
    abs_path: str
    token_count: int


class TrainingDataLoader:
    '''
    This class will be in charge of loading data from disk into memory while
    training. It'll have the ability to work with various shards of training data,
    account for multi-GPU training and perform batched sequence-level random selection.

    Note this uses a seed for reproducibility.
    '''


    def __init__(self, ddp, dataset_dir: str, batch_count: int, tokens_per_batch: int):
        self.ddp = ddp
        self.rank = ddp.local_rank
        self.world_size = ddp.world_size
        self.curr_batch_count = batch_count
        self.curr_tokens_per_batch = tokens_per_batch
        self.rnd_sample_idx_gen = None

        # Store one validation shard
        self.curr_val_idx = 0
        self.val_shard = None
        # Store training shards, keys will be 0 to dict length - 1.
        # All ranks will have the same key to shard mapping.
        self.train_shards = OrderedDict()

        # Load dataset

        self.dataset_dir = dataset_dir

        assert (
            os.path.isdir(self.dataset_dir) 
            and len(os.listdir(self.dataset_dir)) >= 2 # At least 1 val and 1 train shard
        ), f'Error loading dataset. dataset_dir: {self.dataset_dir}, proc_rank: {self.rank}, world_size: {self.world_size}'

        if self.rank == 0: print(f'Loading dataset from {self.dataset_dir}')

        # Setup train and val splits

        self._setup_splits()

        # Create random key generator for `train_shards` dict
        self.train_shrds_rnd_key_gen = RandIdxSeqGen(
            ddp = self.ddp,
            seqLen=len(self.train_shards),
            rank=self.rank,
            world_size=self.world_size,
            rnd_seed=SHARD_RND_SEED_BASE,
        )
        
        # print(f'DEBUG: B rank: {self.rank}. rnd idx order: {self.train_shrds_rnd_key_gen.rnd_ordered_idx}')

        # Setup training shard to use first
        self._setup_training_shard()

    def get_train_samples(self, batch_count, tokens_per_batch):

        did_update_sampling_len = False

        if self.rnd_sample_idx_gen is None:
            # Setup rnd_sample_idx_gen for the first time
            self._setup_sampling_rnd_idx_gen(batch_count, tokens_per_batch)
            did_update_sampling_len = True

        start_idx = self.rnd_sample_idx_gen.next()

        if start_idx is None:
            # We've exhausted all training samples from this shard, go to next shard
            self._setup_training_shard()
            self._setup_sampling_rnd_idx_gen(batch_count, tokens_per_batch)
            did_update_sampling_len = True
            start_idx = self.rnd_sample_idx_gen.next()

        if (
            not did_update_sampling_len
            and (
                self.curr_batch_count != batch_count
                or self.curr_tokens_per_batch != tokens_per_batch
            )
            and self.rank == 0
        ):
            print(
                f'''WARN: (rank-{self.rank}) Attempting to change batch_count ({batch_count}) and tokens_per_batch '''
                f'''({tokens_per_batch}) in the middle of iterating through a batch. \n'''
                f'''Change will be allowed once new shard is loaded. \n'''
                f'''self.curr_batch_count: {self.curr_batch_count}, self.curr_tokens_per_batch: {self.curr_tokens_per_batch}'''
            )

        return self._get_batched_tokens(start_idx, self.curr_train_tokens)
    
    def _get_batched_tokens(self, start_idx, tokens):
        tokens_per_rank = self.curr_batch_count * self.curr_tokens_per_batch
        start_idx_offset = start_idx * (self.world_size * tokens_per_rank)
        rank_idx_offset = self.rank * tokens_per_rank
        start_idx = start_idx_offset + rank_idx_offset
        end_idx = start_idx + tokens_per_rank + 1

        batch = tokens[start_idx: end_idx]
        inputs = (batch[:-1]).view(self.curr_batch_count, self.curr_tokens_per_batch)
        targets = (batch[1:]).view(self.curr_batch_count, self.curr_tokens_per_batch)

        inputs = inputs.to(self.ddp.assigned_device)
        targets = targets.to(self.ddp.assigned_device)
        return inputs, targets
    
    def reset_validation(self):
        self.curr_val_idx = 0

    def get_val_samples(self):
        sample_window = self.world_size * self.curr_batch_count * self.curr_tokens_per_batch + 1
        if ((self.curr_val_idx + 1) * sample_window) >= len(self.val_shard):
            # Amount of tokens needed does not fit in tokens left
            self.reset_validation()
        inputs, targets = self._get_batched_tokens(self.curr_val_idx, self.val_shard)
        self.curr_val_idx += 1
        return inputs, targets

    def _setup_sampling_rnd_idx_gen(self, batch_count, tokens_per_batch):
        self.curr_batch_count = batch_count
        self.curr_tokens_per_batch = tokens_per_batch
        sample_window_size = self.world_size * self.curr_batch_count * self.curr_tokens_per_batch

        seq_len = int(len(self.curr_train_tokens) / sample_window_size)

        # print(f'DEBUG: seq_len: {seq_len} | len-curr_train_tokens: {len(self.curr_train_tokens)} | sample_window_size: {sample_window_size}')

        if len(self.curr_train_tokens) % sample_window_size == 0:
            # There won't be enough tokens for the target tensors, since targe tensor
            # is always offset ahead by 1 token.
            seq_len -= 1

        if self.rnd_sample_idx_gen is None:
            self.rnd_sample_idx_gen = RandIdxSeqGen(
                ddp = self.ddp,
                seqLen=seq_len,
                rank=self.rank,
                world_size=self.world_size,
                rnd_seed=SAMPLE_RND_SEED_BASE,
            )
        else:
            self.rnd_sample_idx_gen.reset(seq_len)

        # print(f'DEBUG: rnd_sample_idx_gen order: {self.rnd_sample_idx_gen.rnd_ordered_idx}')

    def _setup_splits(self):
        '''
        Setup `self.val_shard`, `self.train_shards`, and self.rand_train_shards_order.
        For now, `self.val_shard` will hold only the first shard in the dataset,
        all the other shards will be held in `self.train_shards`.
        Note that the value of `self.val_shard` will never change.
        '''
        
        # Setup self.val_shard and self.train_shards
        for file_name in os.listdir(self.dataset_dir):
            match = SHARD_RE_PATTERN.search(file_name)
            assert match, f'Filename {file_name} does not match the expected pattern'
            
            idx = int(match.group(1))
            token_count = int(match.group(2))
            file_path = os.path.join(self.dataset_dir, file_name)
            shard = Shard(file_path, token_count)

            if idx == 0:
                # Set validation Shard
                self.val_shard = self._load_np_arr(file_path)
            else:
                # Set all training Shards
                self.train_shards[idx-1] = shard

    def _setup_training_shard(self):
        '''
        Figure out which training shard to use.
        If all have been used, reshuffle their order for the next epoch.
        '''

        next_shrd_key = self.train_shrds_rnd_key_gen.next()

        if next_shrd_key is None:
            # One epoch of data has been completed
            if self.ddp.is_main:
                log.info('Entire training dataset has been seen, will shuffle and iterate through it again.')
            self.train_shrds_rnd_key_gen.reset(len(self.train_shards))
            next_shrd_key = self.train_shrds_rnd_key_gen.next()
            # print(f'DEBUG: train_shrds_rnd_key_gen order: {self.train_shrds_rnd_key_gen.rnd_ordered_idx}')

        shard_file_path = self.train_shards[next_shrd_key].abs_path
        self.curr_train_tokens = self._load_np_arr(shard_file_path)

        if self.ddp.is_main:
            log.info(f'Next shard key to use: {next_shrd_key}. shard_file_path: {shard_file_path}')

    def _load_np_arr(self, npy_path):
        tokens = np.load(npy_path)
        tokens = tokens.astype(np.int32)
        return torch.tensor(tokens, dtype=torch.long)
