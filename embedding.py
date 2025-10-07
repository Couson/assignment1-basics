import torch
from torch.nn.parameter import Parameter

from einops import einsum, rearrange

class MyEmbedding(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings # vocab size
        self.embedding_dim = embedding_dim # model dim
        self.device = device
        self.dtype = dtype

        self.embedding = Parameter(torch.randn((num_embeddings, embedding_dim),
                                                dtype=dtype))
        self.init_parameters()
    
    def init_parameters(self):
        MEAN = 0
        VAR = 1
        STD = VAR ** 0.5
        UPPER, LOWER = 3 * STD, -3 * STD

        torch.nn.init.trunc_normal_(self.embedding,
                                    mean=MEAN,
                                    std=STD,
                                    a=LOWER,
                                    b=UPPER
                                    )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # self.embedding "v b"
        # token_ids "... v"
        return self.embedding[token_ids]