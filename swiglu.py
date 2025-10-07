import torch
from torch.nn.parameter import Parameter

from einops import einsum


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Parameter(torch.empty((self.d_ff, self.d_model)))
        self.w2 = Parameter(torch.empty((self.d_model, self.d_ff)))
        self.w3 = Parameter(torch.empty((self.d_ff, self.d_model)))
        self.silu = SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_w1_x = self.silu(einsum(
            self.w1,
            x,
            "d_ff d_model, ... d_model -> ... d_ff"
        ))

        w3_x = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")

        x = silu_w1_x * w3_x

        return einsum(self.w2, x, "d_model d_ff, ... d_ff -> ... d_model")



class SiLU(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            x,
            torch.sigmoid(x),
            "..., ... -> ..."
        )


