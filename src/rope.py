import logging

import torch
from einops import rearrange, repeat

from src.params import HParams
from src.global_cache import global_cache


BASE_THETA: float = 10_000.0
COS_CACHE_KEY = 'cos_cache'
SIN_CACHE_KEY = 'sin_cache'
logger = logging.getLogger(__name__)


class Rope():
    '''
    Implement Rotary Positional Embeddings (RoPE).
    Splitting the embedding space in half instead of interleaving entries.
    '''

    def __init__(self, hParams: HParams):
        super().__init__()
        self.n_ctx = hParams.n_ctx
        self.head_dim = hParams.n_embd // hParams.n_head
        self._prepare_rotary_embeddings()

    def _prepare_rotary_embeddings(self):
        '''
        Precompute the cosine and sine values for rotary embeddings.
        '''
        assert self.head_dim % 2 == 0, 'Head dimension must be even for RoPE.'

        if COS_CACHE_KEY in global_cache and SIN_CACHE_KEY in global_cache:
            logger.info(f"{COS_CACHE_KEY} and {SIN_CACHE_KEY} have been previously created, skipping creation.")
            return
        else:
            # Compute inverse frequencies
            # Implicitly working with floats for this one-time operation
            half_dim = self.head_dim // 2
            freq_seq = torch.arange(half_dim, dtype=torch.float32)
            inverse_freq = 1.0 / (BASE_THETA ** (freq_seq / half_dim))

            # Compute position indices
            position_seq = torch.arange(self.n_ctx, dtype=torch.float32)
            angle_rates = torch.outer(position_seq, inverse_freq)

            # Stack angles to match total head_dim, shape (n_ctx, head_dim)
            angles = repeat(angle_rates, 'n_ctx half_dim -> n_ctx (2 half_dim)')

            # Store cosine and sine values in cache to share with other modules
            global_cache[COS_CACHE_KEY] = torch.cos(angles)
            global_cache[SIN_CACHE_KEY] = torch.sin(angles)

    def _apply_rotary(self, x):
        '''
        Apply rotary positional embedding to the input tensor.
        '''
        _, _, n_ctx, _ = x.shape
        cos_cached = global_cache[COS_CACHE_KEY][:n_ctx, :].to(x.device)
        sin_cached = global_cache[SIN_CACHE_KEY][:n_ctx, :].to(x.device)

        # Reshape (1, 1, n_ctx, head_dim)
        cos_cached = rearrange(cos_cached, 'n_ctx head_dim -> 1 1 n_ctx head_dim')
        sin_cached = rearrange(sin_cached, 'n_ctx head_dim -> 1 1 n_ctx head_dim')

        # Split the last dimension into two halves (batch_size, n_head, n_ctx, head_dim // 2)
        x_r = rearrange(x, '... (two d) -> ... two d', two=2)
        x1, x2 = x_r.unbind(dim=-2)

        # Apply the rotary embeddings
        x_rotated = torch.cat([-x2, x1], dim=-1)  # shape (batch_size, n_head, n_ctx, head_dim)
        return (x * cos_cached) + (x_rotated * sin_cached)

    def apply_rotary(self, x):
        '''
        x and output shape: (batch_size, n_head, n_ctx, head_dim).
        '''
        assert x.size(-1) == self.head_dim, 'Input head_dim does not match model head_dim.'
        return self._apply_rotary(x)


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    # (Batch Size, Number of Heads, Sequence Length, Head Dimension)
    B, H, S, HD = 2, 3, 2, 8
    # q = torch.empty(B, H, S, HD)
    # q[:,:,:,:] = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    q = torch.tensor([
        [
            [[0.0756, 0.1966, 0.3164, 0.4017, 0.1186, 0.8274, 0.3821, 0.6605],
            [0.8536, 0.5932, 0.6367, 0.9826, 0.2745, 0.6584, 0.2775, 0.8573]],
            [[0.8993, 0.0390, 0.9268, 0.7388, 0.7179, 0.7058, 0.9156, 0.4340],
            [0.0772, 0.3565, 0.1479, 0.5331, 0.4066, 0.2318, 0.4545, 0.9737]],
            [[0.4606, 0.5159, 0.4220, 0.5786, 0.9455, 0.8057, 0.6775, 0.6087],
            [0.6179, 0.6932, 0.4354, 0.0353, 0.1908, 0.9268, 0.5299, 0.0950]]
        ],
        [
            [[0.5789, 0.9131, 0.0275, 0.1634, 0.3009, 0.5201, 0.3834, 0.4451],
            [0.0126, 0.7341, 0.9389, 0.8056, 0.1459, 0.0969, 0.7076, 0.5112]],
            [[0.7050, 0.0114, 0.4702, 0.8526, 0.7320, 0.5183, 0.5983, 0.4527],
            [0.2251, 0.3111, 0.1955, 0.9153, 0.7751, 0.6749, 0.1166, 0.8858]],
            [[0.6568, 0.8459, 0.3033, 0.6060, 0.9882, 0.8363, 0.9010, 0.3950],
            [0.8809, 0.1084, 0.5432, 0.2185, 0.3834, 0.3720, 0.5374, 0.9551]]
        ]
    ])
    
    hParams = HParams(
        n_vocab = 16,
        n_ctx = S,
        n_embd = H * HD,
        n_head = H,
        n_layer = 2,
    )

    r = Rope(hParams)

    # print(f'q shape: {q.shape}')
    # print(f'q: {q}')

    q_rotated = r.apply_rotary(q)

    # print(f'q_rotated shape: {q_rotated.shape}')
    # print(f'q_rotated: {q_rotated}')

    expected_rotated = torch.tensor([
        [
            [[ 0.0756,  0.1966,  0.3164,  0.4017,  0.1186,  0.8274,  0.3821, 0.6605],
             [ 0.2302,  0.5245,  0.6339,  0.9817,  0.8666,  0.7143,  0.2839, 0.8583]],
            [[ 0.8993,  0.0390,  0.9268,  0.7388,  0.7179,  0.7058,  0.9156, 0.4340],
             [-0.3004,  0.3316,  0.1433,  0.5321,  0.2846,  0.2662,  0.4560, 0.9742]],
            [[ 0.4606,  0.5159,  0.4220,  0.5786,  0.9455,  0.8057,  0.6775, 0.6087],
             [ 0.1733,  0.5972,  0.4301,  0.0352,  0.6230,  0.9914,  0.5342, 0.0950]]
        ],
        [
            [[ 0.5789,  0.9131,  0.0275,  0.1634,  0.3009,  0.5201,  0.3834, 0.4451],
             [-0.1160,  0.7208,  0.9318,  0.8051,  0.0894,  0.1697,  0.7170, 0.5120]],
            [[ 0.7050,  0.0114,  0.4702,  0.8526,  0.7320,  0.5183,  0.5983, 0.4527],
             [-0.5306,  0.2422,  0.1943,  0.9144,  0.6082,  0.7026,  0.1185, 0.8867]],
            [[ 0.6568,  0.8459,  0.3033,  0.6060,  0.9882,  0.8363,  0.9010, 0.3950],
             [ 0.1533,  0.0707,  0.5378,  0.2175,  0.9484,  0.3810,  0.5428, 0.9553]]
        ]
    ])

    if torch.equal(torch.round(q_rotated * 10000) / 10000, expected_rotated):
        print('alls good')
    else:
        print('ERROR: mismatch between q_rotated and expected_rotated!')

    # Check that cos/sin is cached
    r2 = Rope(hParams)
