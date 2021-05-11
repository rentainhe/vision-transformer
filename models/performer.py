import math
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Performer(nn.Module):
    def __init__(self, dim, inner_dim, kernel_ratio=0.5, dropout=0.):
        super().__init__()
        self.emb = inner_dim
        self.to_kqv = nn.Linear(dim, 3 * self.emb)
        self.dropout = nn.Dropout(dropout)
        self.project_out = nn.Linear(self.emb, self.emb)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            nn.GELU(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dropout)
        )

        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)
        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def attention(self, x):
        b, n, _ = x.shape
        k, q, v = torch.split(self.to_kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        y = v + self.dropout(self.project_out(y))  # same as token_transformer in T2T layer, use v as skip connection

        return y

    def forward(self, x):
        x = self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x