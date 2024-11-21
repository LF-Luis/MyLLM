import os
import logging
import math
import time

import torch

from src.model import LLM
from src.utils.logger import setup_logging
from src.utils.handle_ddp import DDPHandler
from src.utils.root import get_temp_data_abs_path
from src.model_utils.adamw_opt import AdamWOptimizer
from src.model_utils.debugging import get_model_size, log_training_metrics
from src.model_utils.checkpoint_utils import save_checkpoint
from src.model_configs.my_llm_config import get_llm_config
from src.data_processing.training_data_loader import TrainingDataLoader


'''
Main script to pre-train MyLLM in 8xA100 GPUs.
'''

if __name__ == "__main__":
    ddp = DDPHandler()
    setup_logging()
    log = logging.getLogger(__name__)

    # Set up all parameters
    hParams, tParams = get_llm_config()
    batch_size = tParams.batch_token_count / ddp.world_size / hParams.n_ctx
    assert batch_size.is_integer(), f'batch_size {batch_size} ends up being a float.'
    batch_size = int(batch_size)

    # Setup model and optimizer
    # Make sure to keep this order: move to device, compile, then DDP wrap
    model = LLM(hParams)
    model.to(ddp.assigned_device)
    if ddp.is_avail:  
        model = torch.compile(model)
    model = ddp.wrap_model(model)  # Only wraps if CUDA + GPU is available
    model.train()
    opt = AdamWOptimizer(tParams, ddp, ddp.get_actual_model(model))
    torch.set_float32_matmul_precision('high')  # Enable TF32

    if ddp.is_main:
        log.info(f'Model size (full): {get_model_size(model):,}')
        log.info(f'Model size: {math.ceil(get_model_size(model) / 1_000_000)}M')
        log.info(f'batch_size: {batch_size}')
        log.info(f'hParams: {hParams}')
        log.info(f'tParams: {tParams}')

    # Prep data loader
    data_loader = TrainingDataLoader(
        dataset_dir=os.path.join(get_temp_data_abs_path(), 'edu_fineweb10B'),
        rank=ddp.local_rank,
        world_size=ddp.world_size,
        batch_count=batch_size,
        tokens_per_batch=hParams.n_ctx,
    )

    ddp.barrier()

    for step in range(tParams.tot_steps):
        '''
        Training code.
        '''
        step_start_time = time.time()
        opt.zero_grad()

        input, output = data_loader.get_train_samples(batch_size, hParams.n_ctx)
        input, output = input.to(ddp.assigned_device), output.to(ddp.assigned_device)
        with torch.autocast(device_type=ddp.device_type, dtype=torch.bfloat16):
            _, loss = model(input, output)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tParams.clip_grad_max_norm)
        
        opt.step(step=step)

        '''
        Log metrics and save checkpoints at certain intervals.
        Make to wait for GPU compute be done before logging, and
        sync all distributed processes before checkpointing.
        '''
        is_last_step = (step == (tParams.tot_steps - 1))
        should_log = (step % tParams.logging_interval == 0) or is_last_step
        should_checkpoint = (step in tParams.checkpointing_steps) or is_last_step
        
        if ddp.is_avail and should_log:
            torch.cuda.synchronize()
        if ddp.is_avail and should_checkpoint:
            ddp.barrier()

        if ddp.is_main and should_log:
            log_training_metrics(log, ddp, tParams, step_start_time, step, loss, grad_norm)
        if ddp.is_main and should_checkpoint:
            save_checkpoint(ddp.get_actual_model(model), opt.optimizer, step)

    ddp.barrier()
    ddp.end()
