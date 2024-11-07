import torch
import torch.nn as nn

from src.params import HParams


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
    - Using dropout to help with generalization of small model.
    '''

    def __init__(self, hParams: HParams):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hParams.n_embd, TWO_FOR_ONE * HIDDEN_EMBD_SCALE * hParams.n_embd),
            SwiGLU(),
            nn.Dropout(hParams.ffn_dropout),  # Help ensure that non-linear transformations are regularized
            nn.Linear(HIDDEN_EMBD_SCALE * hParams.n_embd, hParams.n_embd),
        )
        
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
        n_layer = 0.,
        ffn_dropout = 0.1,
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

    expected_output = torch.tensor([[[ 0.1274, -0.0766, -0.1010,  0.1599,  0.0196,  0.0589, -0.1362,
          -0.1856],
         [ 0.1235, -0.1349, -0.1412,  0.0738, -0.0216,  0.1400, -0.0621,
          -0.0961],
         [ 0.0967, -0.1291, -0.1125,  0.1381, -0.0056,  0.1013, -0.0937,
          -0.1164],
         [ 0.1150, -0.1096, -0.1162,  0.0749,  0.0149,  0.1102, -0.0866,
          -0.1101]],

        [[ 0.1510, -0.0952, -0.1174,  0.0986,  0.0182,  0.1172, -0.1104,
          -0.1364],
         [ 0.1071, -0.1371, -0.1242,  0.1373,  0.0156,  0.0754, -0.1098,
          -0.1197],
         [ 0.1486, -0.1094, -0.1264,  0.1391,  0.0187,  0.0935, -0.1690,
          -0.1305],
         [ 0.2138, -0.0309, -0.1391,  0.0895, -0.0202,  0.1135, -0.0517,
          -0.1611]]])
    
    ffn = FFN(hParams)
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
