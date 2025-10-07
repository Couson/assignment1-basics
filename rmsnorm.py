import torch
from torch.nn.parameter import Parameter

from einops import einsum, rearrange


class RMSNorm(torch.nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device 
        self.dtype = dtype

        self.weight = Parameter(torch.ones(self.d_model, dtype=self.dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_dtype = x.dtype
        x = x.to(torch.float32)

        x1 =  torch.div(x, 
                        torch.sqrt(torch.mean(x.pow(2), axis=-1, keepdim=True) + self.eps)
                        )
        out = einsum(
            x1, self.weight,
            "... d, d -> ... d"
        )

        return out.to(x_dtype)
