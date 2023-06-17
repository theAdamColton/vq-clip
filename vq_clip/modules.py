import torch
from torch import nn
from torch.nn import functional as F


class Bottleneck(nn.Sequential):
    def __init__(self, in_dim, hidden_dim):
        c_fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        c_fc2 = nn.Linear(in_dim, hidden_dim, bias=False)
        super().__init__(*[c_fc1, c_fc2])


class Block(nn.Module):
    def __init__(self, in_dim, hidden_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = MLP(in_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mlp(self.norm(x))
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.c_fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, in_dim, bias=False)

    def forward(self, x: torch.Tensor):
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x
