import torch
from datetime import datetime
import logging

from src.utils.root import create_temp_data_file

'''
Utility functions to save and load model checkpoint.
All info needed to continue training the model is saved in the checkpoint.
'''

log = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, step, scheduler=None, save_random_state=True):
    #TODO: Needs to save the state of the training_data_loader.py as well.
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,  # Might use this later
        'step': step,
        'hParams': model.hParams,
    }
    
    # Optionally save the random state
    if save_random_state:
        checkpoint['random_state'] = torch.get_rng_state()
        if torch.cuda.is_available():
            checkpoint['cuda_random_state'] = torch.cuda.get_rng_state_all()
    else:
        checkpoint['random_state'] = None
        checkpoint['cuda_random_state'] = None

    date = datetime.now().astimezone()
    date = date.strftime('%Y_%m_%d-%H_%M_%Z')
    filepath = create_temp_data_file('checkpoints', f'checkpoint_step_{step}_date_{date}.pth')
    torch.save(checkpoint, filepath)
    log.info(f"Checkpoint saved at step {step}.")


def load_checkpoint(model, optimizer=None, scheduler=None, filepath='checkpoint.pth', load_random_state=True):
    checkpoint = torch.load(filepath, map_location="cuda" if torch.cuda.is_available() else "cpu")
    # log.info(f'Loading checkpoint with data: {checkpoint}')
    
    # Load model and optimizer states
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if available
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Optionally load the random state
    if load_random_state and checkpoint.get('random_state') is not None:
        torch.set_rng_state(checkpoint['random_state'])
        if torch.cuda.is_available() and checkpoint.get('cuda_random_state') is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_random_state'])

    step = checkpoint['step']
    # log.info(f"Checkpoint loaded, resuming training from step {step}.")

    return step, checkpoint['hParams']
