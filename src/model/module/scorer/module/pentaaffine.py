import torch.nn as nn
from supar.modules import  MLP
import torch
class DecomposedPentaAffineScorer(nn.Module):
    def __init__(self, n_in=800, n_out=400, dropout=0.33, rank=500, scaling=False):
        super(DecomposedPentaAffineScorer, self).__init__()
        self.l = MLP(n_in=n_in, n_out=n_out, dropout=dropout)
        self.m = MLP(n_in=n_in, n_out=n_out, dropout=dropout)
        self.r = MLP(n_in=n_in, n_out=n_out, dropout=dropout)

        self.label_1 = MLP(n_in=n_in, n_out=n_out, dropout=dropout)
        self.label_2 = MLP(n_in=n_in, n_out=n_out, dropout=dropout)

        self.W_1 = nn.Parameter(torch.randn(n_out+1, rank))
        self.W_2 = nn.Parameter(torch.randn(n_out+1, rank))
        self.W_3 = nn.Parameter(torch.randn(n_out+1, rank))
        self.W_4 = nn.Parameter(torch.randn(n_out+1, rank))
        self.W_5 = nn.Parameter(torch.randn(n_out+1, rank))

        self._init()
        self.scaling = scaling
        self.rank = rank

    def _init(self):
        nn.init.zeros_(self.W_1)
        nn.init.zeros_(self.W_2)
        nn.init.zeros_(self.W_3)
        nn.init.zeros_(self.W_4)
        nn.init.zeros_(self.W_5)

    def forward(self, h, l):
        left = self.l(h)
        mid = self.m(h)
        right = self.r(h)
        label_1 = self.label_1(l)
        label_2 = self.label_2(l)
        left = torch.cat((left, torch.ones_like(left[..., :1])), -1)
        mid = torch.cat((mid, torch.ones_like(mid[..., :1])), -1)
        right = torch.cat((right, torch.ones_like(right[..., :1])), -1)
        label_1 = torch.cat((label_1, torch.ones_like(label_1[..., :1])), -1)
        label_2 = torch.cat((label_2, torch.ones_like(label_2[..., :1])), -1)
        left = left @ self.W_1
        mid = mid @ self.W_2
        right = right@ self.W_3
        label_1 = label_1@ self.W_4
        label_2 = label_2@self.W_5
        # null label. no factor.
        label_1[..., 0] = 0
        label_2[..., 0] = 0

        if self.scaling:
            left /= (self.rank ** 1/5)
            right /= (self.rank ** 1/5)
            mid /= (self.rank ** 1/5)
            label_1 /= (self.rank ** 1/5)
            label_2 /= (self.rank ** 1/5)

        return left, mid, right, label_1, label_2


