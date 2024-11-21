import os
import logging

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


log = logging.getLogger(__name__)

WORLD_SIZE = 'WORLD_SIZE'
LOCAL_RANK = 'LOCAL_RANK'
NCCL = 'nccl'
CUDA = "cuda"
MPS = "mps"
CPU = "cpu"


class DDPHandler:
    '''
    Helper class to handle PyTorch's Distributed Data Parallel (DDP) states.
    Note, training with at most one node (with N-number of GPUs, usually 4 virtual or 8).
    '''

    def __init__(self):
        self.is_avail = self._check_ddp_availability()

        if self.is_avail:
            dist.init_process_group(backend=NCCL)
            self.local_rank = int(os.environ[LOCAL_RANK])
            self.world_size = int(os.environ[WORLD_SIZE])
            self.is_main = self.local_rank == 0
            self.assigned_device = f'{CUDA}:{self.local_rank}'
            self.device_type = CUDA
            torch.cuda.set_device(self.assigned_device)
        else:
            self.local_rank = 0
            self.world_size = 1
            self.is_main = True
            if torch.cuda.is_available():
                device = CUDA
            elif hasattr(torch.backends, MPS) and torch.backends.mps.is_available():
                device = MPS
            else:
                device = CPU
            self.assigned_device = device
            self.device_type = CUDA if device.startswith(CUDA) else CPU

        log.info((
            f'Launching worker with config: \n'
            f'local_rank: {self.local_rank} | world_size: {self.world_size} | is_main: {self.is_main} \n'
            f'assigned_device: {self.assigned_device} | device_type: {self.device_type}.'
        ))

    def _check_ddp_availability(self):
        return (
            int(os.environ.get(WORLD_SIZE, 1)) > 1 and  # WORLD_SIZE > 1 indicates DDP is intended
            (dist.is_nccl_available() or dist.is_gloo_available()) and  # Required backend
            torch.cuda.is_available() and  # Must be available
            torch.cuda.device_count() > 1  # At least two GPUs
        )

    def wrap_model(self, model):
        '''
        Only wrap if GPU + CUDA is available.
        Single node, single GPU, no need to use `output_device` for now
        '''
        if self.is_avail:
            model = DDP(model, device_ids=[self.local_rank])
        return model

    def get_actual_model(self, model):
        # Returns the actual model, whether it's wrapped in DDP or not.
        if isinstance(model, DDP):
            return model.module
        return model

    def barrier(self):
        # Synchronize all processes, only used when running with DDP.
        if self.is_avail:
            dist.barrier()

    def end(self):
        '''
        Required to run at the end of training.
        '''
        if self.is_avail:
            if self.is_main:
                log.info(f'Closing global process group.')
            dist.destroy_process_group()
        else:
            pass


if __name__ == "__main__":
    # torchrun --standalone --nproc_per_node=8 src/utils/handle_ddp.py

    logging.basicConfig(level=logging.DEBUG)

    ddp = DDPHandler()
    ddp.end()
