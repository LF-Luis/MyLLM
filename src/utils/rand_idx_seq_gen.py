import torch
import torch.distributed as dist


class RandIdxSeqGen:
    '''
    Random Index Sequence Generator.
    Class to create a sequence of indices that have been randomly shuffled.
    Class allows for the indices to be updated.

    For example, for a given sequence length `seqLen` of 6, which would represent a
    list of 6 items, we would have the following indices:
    [0, 1, 2, 3, 4, 5]

    This class would shuffle the indices and give you access to the current and next index:
    shuffled: [5, 3, 2, 4, 0, 1]
                     ↑  ↑
               current  next

    Set `rank` and `world_size` so that the random order created by rank=0 is shared to
    all ranks.
    '''

    def __init__(self, ddp, seqLen, rank=None, world_size=None, rnd_seed=None):
        self.ddp = ddp
        self.seqLen = seqLen
        self.rank = rank
        self.world_size = world_size
        self.rnd_seed = rnd_seed
        self.generator = torch.Generator()
        self.rnd_ordered_idx = None
        self.current_pointer = None
        self.reset(self.seqLen)

    def reset(self, seqLen):
        assert self.seqLen > 0
        
        self.seqLen = seqLen
        self.current_pointer = None

        if self.rnd_seed:
            self.generator.manual_seed(self.rnd_seed)
            self.rnd_seed += 1

        self.rnd_ordered_idx = torch.randperm(n=self.seqLen, generator=self.generator)

        if (
            self.rank is not None
            and self.world_size is not None
            and self.world_size > 1
        ):
            if self.ddp.is_avail:
                self.rnd_ordered_idx = self.rnd_ordered_idx.to(self.ddp.assigned_device)
            dist.broadcast(self.rnd_ordered_idx, src=0)

    def next(self):
        '''
        Returns next random index value.
        If all values have been exhausted, this will return None until the `reset()` is run.
        '''
        assert (
            self.rnd_ordered_idx is not None and self.rnd_ordered_idx.numel() > 0
        ), f'ERROR: Calling next() before setting self.rnd_ordered_idx: {self.rnd_ordered_idx}'

        if self.current_pointer is None:
            self.current_pointer = 0
        else:
            self.current_pointer += 1

        if self.current_pointer >= self.rnd_ordered_idx.size(0,):
            self.current_pointer -= 1
            return None

        return self.current()

    def current(self):
        if self.current_pointer is None:
            return None
        return int(self.rnd_ordered_idx[self.current_pointer])
