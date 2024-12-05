import logging

import torch
import torch.distributed as dist


log = logging.getLogger(__name__)

class Validation:
    '''
    Module to test validation data set as the model is learning.
    Will test over the same dataset every time.
    '''

    def __init__(self, model, data_loader, tParams, ddp):
        self.model = model
        self.data_loader = data_loader
        self.ddp = ddp
        self.validation_steps = tParams.validation_steps

    def run_validation(self, step):
        '''
        Returns validation loss
        '''
        self.model.eval()
        self.data_loader.reset_validation()

        val_loss = 0.0
    
        with torch.no_grad():
            for _ in range(self.validation_steps):
                input, output = self.data_loader.get_val_samples()
                with torch.autocast(device_type=self.ddp.device_type, dtype=torch.bfloat16):
                    _, loss = self.model(input, output)
                val_loss += loss.detach()

        if self.ddp.is_avail:
            # Reduce loss across all processes
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        
        self.model.train()

        # Average over validation steps
        val_loss = val_loss / self.validation_steps

        if self.ddp.is_main:
            log.info(f'Step ({step}). Val Loss: {val_loss.item():.4f}')
