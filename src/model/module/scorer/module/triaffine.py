import torch.nn as nn
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


# class TriaffineScorer
# class TriaffineScorer(nn.Module):





