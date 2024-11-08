import torch
import torch.nn as nn

from src.params import HParams
from src.attention import Attention
from src.ffn import FFN


class TransformerBlock(nn.Module):

    def __init__(self, hParams: HParams):
        '''
        - Using pre-layer normalization to improve training stability in deep network,
        allowing for faster convergence.
        - Using RMSNorm for its efficient (lower compute/memory w.r.t LayerNorm), stable normalization
        with the limited pre-training dataset and small LLM.
        '''
        super().__init__()
        self.attn = Attention(hParams)  # Casual, multi-head attention module.
        self.ffn = FFN(hParams)
        self.norm1 = nn.RMSNorm(hParams.n_embd, eps=1e-5)  # Test later: scale=True)
        self.norm2 = nn.RMSNorm(hParams.n_embd, eps=1e-5)
        self.attn_dropout = nn.Dropout(hParams.attn_res_pdrop)
        '''
        Will rely on dropout post SwiGLU() activation, where the hidden space is larger. This
        should encourage the model to develop more robust features, becoming less 
        overly reliant on specific neurons
        '''
        # self.ffn_dropout = nn.Dropout(hParams.ffn_res_pdrop)
        
    def forward(self, x):
        xn = self.norm1(x)
        x = x + self.attn_dropout(self.attn(xn))
        xn = self.norm2(x)
        # return x + self.ffn_dropout(self.ffn(xn))
        return x + self.ffn(xn)
    

if __name__ == '__main__':
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    
    hParams = HParams(
        n_vocab = 0.,
        n_ctx = 4,
        n_embd = 8,
        n_head = 2,
        n_layer = 0.,
        ffn_act_pdrop = 0.1,
    )
    
    batch_size, n_ctx, embed_dim = 2, hParams.n_ctx, hParams.n_embd

    x = torch.tensor([
        [[0.0975, 0.2956, 0.9027, 0.3112, 0.9167, 0.4139, 0.4362, 0.6996],
         [0.4265, 0.4958, 0.8463, 0.6671, 0.4801, 0.6904, 0.9355, 0.6260],
         [0.3534, 0.6638, 0.4563, 0.1091, 0.3069, 0.7274, 0.5164, 0.6845],
         [0.2073, 0.9727, 0.2913, 0.6066, 0.2557, 0.2588, 0.7239, 0.3604]],
        [[0.1829, 0.2956, 0.8646, 0.8010, 0.8044, 0.0733, 0.7355, 0.6248],
         [0.1638, 0.5158, 0.6000, 0.2299, 0.2890, 0.9078, 0.4596, 0.4947],
         [0.1836, 0.2010, 0.9603, 0.6861, 0.4209, 0.8046, 0.2621, 0.0638],
         [0.0036, 0.7032, 0.3051, 0.8070, 0.9271, 0.6647, 0.9296, 0.3848]]
    ])

    expected_output = torch.tensor([[[ 0.1519,  0.0489,  1.3749,  0.5524,  1.1652, -0.6570,  0.5610,
           0.5494],
         [ 0.5295,  0.3981,  1.2366,  0.9228,  0.7882, -0.4821,  1.1015,
           0.6673],
         [ 0.3088,  0.5734,  0.8587,  0.4106,  0.5161, -0.4679,  0.6630,
           0.7092],
         [ 0.2552,  0.8746,  0.5995,  1.0866,  0.5140, -0.9584,  0.9113,
           0.4282]],

        [[ 0.4376,  0.1536,  1.2488,  0.8525,  1.2652, -0.9327,  1.0190,
           0.5315],
         [ 0.2235,  0.4767,  0.9094,  0.4402,  0.5326, -0.1893,  0.5899,
           0.4755],
         [ 0.4613,  0.1920,  1.2937,  1.0196,  0.8376, -0.2501,  0.4573,
           0.0873],
         [ 0.2204,  0.6480,  0.5920,  1.0773,  1.1199, -0.3552,  1.1196,
           0.3668]]])
    
    block = TransformerBlock(hParams)
    output = block(x)
    output = torch.round(output * 10000) / 10000

    # print(f'output: {output}')
    
    if torch.equal(output, expected_output):
        print('alls good')
    else:
        not_equal = output != expected_output
        different_indices = not_equal.nonzero(as_tuple=True)
        for idx in zip(*different_indices):
            print(f"Diff at index {idx}: output = {output[idx]}, expected_output = {expected_output[idx]}")
