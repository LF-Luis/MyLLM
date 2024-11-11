import torch
import torch.nn as nn

from src.params import HParams
from src.model_utils.weight_init import init_linear, init_linear_res_proj


HIDDEN_EMBD_SCALE = 4
TWO_FOR_ONE = 2  # Use one linear layer to calculate the output needed from two linear layers


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = nn.SiLU()

    def forward(self, x):
        x1, x2 = x.chunk(TWO_FOR_ONE, dim=-1)
        return x1 * self.silu(x2)


class FFN(nn.Module):
    '''
    Feed-forward neural network. Part of the transformer block, this will perform a
    non-linear transformation to capture complex patterns from the attention mechanism.
    - Keeping bias term in linear layers to help the small model's capacity to learn and generalize.
    - Using dropout to avoid overfitting in this large parameter space of this small model. Let's
    force the network to learn redundant representations, which should improve its 
    generalization capabilities
    '''

    def __init__(self, hParams: HParams):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hParams.n_embd, TWO_FOR_ONE * HIDDEN_EMBD_SCALE * hParams.n_embd),
            SwiGLU(),
            nn.Dropout(hParams.ffn_act_pdrop),
            nn.Linear(HIDDEN_EMBD_SCALE * hParams.n_embd, hParams.n_embd),
        )
        self.reset_parameters(hParams)

    def reset_parameters(self, hParams: HParams):        
        init_linear(self.net[0], hParams)
        init_linear_res_proj(self.net[-1], hParams)
        
    def forward(self, x):
        '''
        x and output: (batch_size, n_ctx, n_embd)
        '''
        return self.net(x)
        

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
        [[ 0.0031,  0.0024,  0.0400,  0.0311,  0.0209,  0.0057, -0.0173, -0.0096],
         [-0.0224, -0.0248,  0.0107,  0.0238, -0.0034,  0.0012, -0.0131, -0.0165],
         [-0.0076, -0.0107,  0.0237,  0.0059,  0.0057,  0.0005, -0.0060, -0.0131],
         [-0.0216, -0.0077, -0.0132,  0.0093,  0.0035,  0.0114,  0.0076, -0.0013]],
        [[-0.0140,  0.0252,  0.0317,  0.0114,  0.0021,  0.0078,  0.0011, -0.0034],
         [ 0.0008, -0.0164,  0.0071,  0.0168,  0.0116, -0.0020, -0.0088, -0.0031],
         [-0.0047, -0.0136, -0.0043,  0.0080, -0.0085, -0.0126, -0.0115, 0.0066],
         [-0.0292, -0.0165,  0.0083,  0.0125,  0.0137,  0.0162,  0.0123, -0.0131]]])
    
    ffn = FFN(hParams)    
    # from utils.debugging import get_stats
    # get_stats(ffn.net[0].weight.data)
    # get_stats(ffn.net[3].weight.data)
    output = ffn(x)
    output = torch.round(output * 10000) / 10000

    # print(f'output: {output}')
    
    if torch.equal(output, expected_output):
        print('alls good')
    else:
        not_equal = output != expected_output
        different_indices = not_equal.nonzero(as_tuple=True)
        for idx in zip(*different_indices):
            print(f"Diff at index {idx}: output = {output[idx]}, expected_output = {expected_output[idx]}")
