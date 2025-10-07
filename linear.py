import torch
from torch.nn.parameter import Parameter

from einops import einsum


class MyLinear(torch.nn.Module):

    def __init__(self, in_feat, out_feat, device=None, dtype=None):
        super().__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.device = device
        self.dtype = dtype

        # TODO
        self.weight = Parameter(torch.empty((out_feat, in_feat), dtype=self.dtype))
        self.init_parameters()

    def init_parameters(self):
        MEAN = 0
        VAR = 2 / (self.in_feat + self.out_feat)
        STD = VAR ** 0.5
        UPPER, LOWER = 3 * STD, -3 * STD

        torch.nn.init.trunc_normal_(self.weight, 
                                    mean=MEAN,
                                    std=STD,
                                    a=LOWER,
                                    b=UPPER
                                    )

    def forward(self, x: torch.Tensor):
        return einsum(
            x, # "... in""
            self.weight, # "out in"
            "... in, out in -> ... out" 
        )

# if __name__ == "__main__":

