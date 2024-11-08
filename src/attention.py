import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from src.params import HParams
from src.rope import Rope


class Attention(nn.Module):
    '''
    Casual, multi-head attention module.
    - Using Rotary Positional Embeddings (RoPE) since it can help small models by efficiently encoding relative positions, improving contextual understanding and token relationships even in limited contexts, like 2048 tokens.
    - Not using Grouped Query Attention (GQA) since we're training a small language model with a small sequence length. GQA primarily enhances efficiency in large models by reducing computational complexity, but its benefits mey be limited for models under 1B parameters.
    - Keeping bias term in linear layers to help the small model's capacity to learn and generalize.
    '''

    def __init__(self, hParams: HParams):
        super().__init__()
        assert hParams.n_embd % hParams.n_head == 0, 'n_embd must be divisible by n_head'
        self.n_embd = hParams.n_embd
        self.n_head = hParams.n_head
        self.head_dim = self.n_embd // self.n_head  # Dimension per head
        # self.attn_pdrop = hParams.attn_pdrop
        self.rope = Rope(hParams)
        self.qkv_proj = nn.Linear(self.n_embd, 3 * self.n_embd)  # Project to Q, K, V
        self.out_proj = nn.Linear(self.n_embd, self.n_embd)  # Output projection

    def forward(self, x):
        batch_size, n_ctx, embed_dim = x.size()
        assert embed_dim == self.n_embd, (
            f'Expected embedding dimension {self.n_embd}, got {embed_dim}'
        )

        '''
        # # (batch_size, n_ctx, 3 * n_embd)
        # qkv = self.qkv_proj(x)
        
        # # Reshape qkv to (batch_size, n_ctx, 3, n_head, head_dim)
        # qkv = qkv.view(batch_size, n_ctx, 3, self.n_head, self.head_dim)
        
        # # Permute to (batch_size, n_head, n_ctx, 3, head_dim)
        # qkv = qkv.permute(0, 3, 1, 2, 4)
        # .contiguous()

        qkv = self.qkv_proj(x).view(batch_size, n_ctx, 3, self.n_head, self.head_dim).permute(0, 3, 1, 2, 4)
        '''

        # (batch_size, n_ctx, 3 * n_embd)
        qkv = self.qkv_proj(x)
        qkv = rearrange(qkv, 'b t (three h d) -> b h t three d', three=3, h=self.n_head)

        # Shape: (batch_size, n_head, n_ctx, head_dim)
        xq, xk, xv = qkv.unbind(dim=3)

        # Apply rotary embeddings
        xq = self.rope.apply_rotary(xq)
        xk = self.rope.apply_rotary(xk)
        
        y = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            is_causal=True,
            # dropout_p=self.attn_pdrop,  # can lead to underfitting and training instability, specially since dropout is being in multiple other places
        )
        
        # Reshape the output back to (batch_size, n_ctx, embed_dim)
        y = y.transpose(1, 2).reshape(batch_size, n_ctx, embed_dim)
        return self.out_proj(y)


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
        [[ 0.0942, -0.3012,  0.4149,  0.0168,  0.1095, -0.5339,  0.0681, 0.0089],
         [ 0.1005, -0.2839,  0.3736,  0.0691,  0.1328, -0.6172,  0.0480, 0.0559],
         [ 0.0723, -0.2863,  0.3848,  0.1243,  0.1238, -0.5957,  0.0631, 0.0658],
         [ 0.0783, -0.2749,  0.3680,  0.1338,  0.1514, -0.5400,  0.0961, 0.0455]],
        [[ 0.1653, -0.3040,  0.3642, -0.0708,  0.2208, -0.4727,  0.1149, -0.0151],
         [ 0.1264, -0.2927,  0.3560,  0.0318,  0.1349, -0.4994,  0.0672, 0.0185],
         [ 0.1567, -0.2911,  0.3213,  0.0021,  0.0895, -0.4415,  0.0377, -0.0077],
         [ 0.1333, -0.2480,  0.3435,  0.0351,  0.1369, -0.4808,  0.0671, -0.0041]]
    ])
    
    casual_self_attention = Attention(hParams)
    output = casual_self_attention(x)
    output = torch.round(output * 10000) / 10000

    # print(f'x shape: {x.shape}')
    # print(f'x: {x}')
    # print(f'output shape: {output.shape}')
    # print(f'output: {output}')
    
    if torch.equal(output, expected_output):
        print('alls good')
    else:
        not_equal = output != expected_output
        different_indices = not_equal.nonzero(as_tuple=True)
        for idx in zip(*different_indices):
            print(f"Diff at index {idx}: output = {output[idx]}, expected_output = {expected_output[idx]}")
