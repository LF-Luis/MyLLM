

'''
Main script to train language model.
Let's pretrain a small (~1B params) model and then finetune it for general knowledge Q&A.
'''



hParams = HParams(
    n_vocab = 50257,  # Vocab size of r50k_base
    n_ctx = 2048,  # context size
    n_embd = 768,  # embedding dimension
    n_head = 12,  # number of attention heads
    n_layer = 12,  # number of attention blocks
    ffn_dropout = 0.1  # dropout rate
)
 
tTParams = TParams(
    max_steps = 19073  # Max number of training steps
)
    
    
    
    