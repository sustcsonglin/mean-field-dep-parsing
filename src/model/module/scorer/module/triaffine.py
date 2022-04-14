import torch.nn as nn
import torch
from supar.modules import  MLP, Triaffine

class TriaffineScorer(nn.Module):
    def __init__(self, n_in=800, n_out=400, n_out_label=1, bias_x=True, bias_y=False, dropout=0.33):
        super(TriaffineScorer, self).__init__()
        self.l = MLP(n_in=n_in, n_out=n_out, dropout=dropout)
        self.m = MLP(n_in=n_in, n_out=n_out, dropout=dropout)
        self.r = MLP(n_in=n_in, n_out=n_out, dropout=dropout)
        self.attn = Triaffine(n_in=n_out, bias_x=bias_x, bias_y=bias_y, n_out=n_out_label)
        self.n_out_label = n_out_label


    def forward(self, h):
        left = self.l(h)
        mid =  self.m(h)
        right = self.r(h)

        #sib, dependent, head)
        if self.n_out_label == 1:
            return self.attn(left, mid, right).permute(0, 2, 3, 1)
        else:
            return self.attn(left, mid, right).permute(0, 2, 3, 4, 1)




class DecomposedTriAffineScorer(nn.Module):
    def __init__(self, n_in=800, n_out=400, dropout=0.33, rank=500, scaling=False):
        super(DecomposedTriAffineScorer, self).__init__()
        self.l = MLP(n_in=n_in, n_out=n_out, dropout=dropout)
        self.m = MLP(n_in=n_in, n_out=n_out, dropout=dropout)
        self.r = MLP(n_in=n_in, n_out=n_out, dropout=dropout)

        self.W_1 = nn.Parameter(torch.randn(n_out + 1, rank))
        self.W_2 = nn.Parameter(torch.randn(n_out + 1, rank))
        self.W_3 = nn.Parameter(torch.randn(n_out + 1, rank))

        self._init()

        self.rank = rank

    def _init(self):
        nn.init.zeros_(self.W_1)
        nn.init.zeros_(self.W_2)
        nn.init.zeros_(self.W_3)

    def forward(self, h):
        left = self.l(h)
        mid = self.m(h)
        right = self.r(h)
        left = torch.cat((left, torch.ones_like(left[..., :1])), -1)
        mid = torch.cat((mid, torch.ones_like(mid[..., :1])), -1)
        right = torch.cat((right, torch.ones_like(right[..., :1])), -1)
        left = left @ self.W_1
        mid = mid @ self.W_2
        right = right @ self.W_3

        return left, mid, right


