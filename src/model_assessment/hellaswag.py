# import os
import logging

import tiktoken
from datasets import load_dataset
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence


from src.utils.root import create_temp_data_dir


log = logging.getLogger(__name__)

EVAL_DIR = "eval"
HELLASWAG_KEY = "hellaswag"
# NUM_PROC_FOR_DOWNLOAD = int(0.75 * os.cpu_count())


class HellaSwag:
    '''
    Class to encapsulate loading HellaSwag validation dataset and
    running evaluation on MyLLM.

    Got the idea from Karpathy's evaluation of build-nanogpt, but prefer the implementation
    from Tenstorrent's Benchmarking, though my implementation is a bit more straightforward since
    I'm only doing HellaSwag eval.

    https://github.com/tenstorrent/benchmarking/blob/main/benchmark/models/falcon/falcon.py#L170
    https://github.com/karpathy/build-nanogpt/blob/master/hellaswag.py
    '''

    def __init__(self, model, tParams, ddp):
        self.ddp = ddp
        self.model = model
        self.tokenizer = tiktoken.get_encoding("r50k_base")

        val_set_key = tParams.eval_hs_subset_key
        eval_dir = create_temp_data_dir(EVAL_DIR)

        # Only one worker should download data, all other workers should wait
        if self.ddp.is_main:
            # TODO: Store tokenized data on disk
            load_dataset(
                HELLASWAG_KEY, 
                split=val_set_key, 
                cache_dir=eval_dir, 
                trust_remote_code=True,
                # num_proc=NUM_PROC_FOR_DOWNLOAD,  # "validation" only contains 1 shard, not needed
            )
        self.ddp.barrier()

        # Load from disk and shard
        self.eval_dataset = load_dataset(
            HELLASWAG_KEY, 
            split=val_set_key, 
            cache_dir=eval_dir,
            trust_remote_code=True,
        ).shard(
            num_shards=self.ddp.world_size,
            index=self.ddp.local_rank,
        )

        log.info(f'Rank: {self.ddp.local_rank}. HellaSwag dataset size: {len(self.eval_dataset)}.')

    def run_eval(self, step):
        self.model.eval()
        self._run_eval(step)
        self.model.train()

    def _run_eval(self, step):
        total_correct = 0
        total_examples = 0

        for example in self.eval_dataset:
            context = example["ctx"]
            endings = example["endings"]
            label = int(example["label"])
            context_ids = self.tokenizer.encode(context)
            context_len = len(context_ids)
            input_ids = []
            ending_masks = []

            for ending in endings:
                ending_ids = self.tokenizer.encode(f" {ending}")
                sample_ids = context_ids + ending_ids
                input_ids.append(torch.tensor(sample_ids))
                ending_masks.append(
                    torch.ones(len(ending_ids), dtype = torch.int)
                )

            # Pad input_ids and ending_masks
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
            input_ids = input_ids.to(self.ddp.assigned_device)
            ending_masks = pad_sequence(ending_masks, batch_first=True, padding_value=0)
            ending_masks = ending_masks.to(self.ddp.assigned_device)

            with torch.no_grad():
                with torch.autocast(device_type=self.ddp.device_type, dtype=torch.bfloat16):
                    #TODO: Batch more samples at once
                    logits, _ = self.model(input_ids)

            # Get logits that belong to the generated 'end' only, and align with corresponding class labels
            logits = logits[:, (context_len-1):-1, :]
            labels = input_ids[:, context_len:]

            # Grab the log-probabilities of the target tokens (labels)
            log_probs = F.log_softmax(logits, dim=-1)
            labels = labels.unsqueeze(-1)
            targets_log_prob = torch.gather(log_probs, dim=-1, index=labels).squeeze(-1)
            targets_log_prob = targets_log_prob * ending_masks.float()

            # Select gen ending with the highest average log-likelihood
            avg_targets_log_prob = targets_log_prob.sum(dim=-1) / ending_masks.sum(dim=-1)
            predicted_label = torch.argmax(avg_targets_log_prob).item()

            if predicted_label == label:
                total_correct += 1
            total_examples += 1

        if self.ddp.is_avail:
            total_correct = torch.tensor(total_correct, device=self.ddp.assigned_device)
            total_examples = torch.tensor(total_examples, device=self.ddp.assigned_device)
            dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_examples, op=dist.ReduceOp.SUM)        
            total_correct = total_correct.item()
            total_examples = total_examples.item()
        accuracy = total_correct / total_examples
        
        if self.ddp.is_main:
            log.info(f"Step ({step}). HellaSwag Evaluation Accuracy: {total_correct}/{total_examples} = {accuracy * 100:.2f}%")
