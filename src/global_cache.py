from typing import Dict
import torch


class GlobalCache(Dict[str, torch.Tensor]):
    '''
    Share repeated, pre-computed data across modules.
    '''
    pass


global_cache = GlobalCache()
