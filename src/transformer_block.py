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
        n_layer = 6,
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

    expected_output = torch.tensor([
        [[ 0.1222,  0.3944,  0.9081,  0.4116,  0.9019,  0.4357,  0.5854, 0.5876],
         [ 0.4480,  0.6237,  0.8530,  0.8256,  0.4027,  0.7042,  1.0869, 0.5828],
         [ 0.4034,  0.7573,  0.4909,  0.2609,  0.2750,  0.7530,  0.7240, 0.6498],
         [ 0.2345,  0.9932,  0.4020,  0.7886,  0.1864,  0.2969,  0.9419, 0.3768]],
        [[ 0.2539,  0.3301,  0.8965,  0.9369,  0.6914,  0.0726,  0.8460, 0.4512],
         [ 0.2245,  0.6699,  0.6241,  0.3642,  0.2603,  0.9434,  0.6830, 0.4444],
         [ 0.2377,  0.3256,  0.9132,  0.7391,  0.3647,  0.8737,  0.4459, -0.0557],
         [ 0.0723,  0.8872,  0.3468,  0.9490,  0.8377,  0.6990,  1.0981, 0.3463]]
    ])
    
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
