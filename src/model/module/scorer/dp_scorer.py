import torch.nn as nn
from .module.biaffine import BiaffineScorer
from .module.triaffine import DecomposedTriAffineScorer
from .module.pentaaffine import DecomposedPentaAffineScorer
import torch
import logging
log = logging.getLogger(__name__)


class CPD_SecondOrderDPScorer(nn.Module):
    def __init__(self, conf, fields, input_dim):
        super(CPD_SecondOrderDPScorer, self).__init__()
        self.conf = conf

        self.rel_scorer = BiaffineScorer(n_in=input_dim, n_out=conf.n_mlp_rel, bias_x=True, bias_y=True,
                                         dropout=conf.mlp_dropout, n_out_label=fields.get_vocab_size("rel"),
                                         scaling=conf.scaling)

        self.label = nn.Parameter(torch.randn(fields.get_vocab_size("rel"), input_dim))
        self.sib_scorer = DecomposedPentaAffineScorer(n_in=input_dim, n_out=conf.n_mlp_rel,
                                          dropout=conf.mlp_dropout, rank=conf.rank, scaling=self.conf.scaling)

        # self.cop_scorer = DecomposedPentaAffineScorer(n_in=input_dim, n_out=conf.n_mlp_rel,
        #                                   dropout=conf.mlp_dropout, rank=conf.rank, scaling=self.conf.scaling)

        self.grd_scorer = DecomposedPentaAffineScorer(n_in=input_dim, n_out=conf.n_mlp_rel,
                                          dropout=conf.mlp_dropout, rank=conf.rank, scaling=self.conf.scaling)

        self.grd_scorer2 = DecomposedPentaAffineScorer(n_in=input_dim, n_out=conf.n_mlp_rel,
                                          dropout=conf.mlp_dropout, rank=conf.rank, scaling = self.conf.scaling)

    def forward(self, ctx):
        x = ctx['encoded_emb'][:, :-1]
        s_rel = self.rel_scorer(x)
        s_rel[..., 0] = 0
        ctx['s_rel'] = s_rel
        ctx['s_sib_a'], ctx['s_sib_b'], ctx['s_sib_c'],  ctx['s_sib_l1'], ctx['s_sib_l2'] = self.sib_scorer(x, self.label)
        # ctx['s_cop_a'], ctx['s_cop_b'], ctx['s_cop_c'], ctx['s_cop_l1'], ctx['s_cop_l2'] = self.cop_scorer(x, self.label)
        ctx['s_grd_a'], ctx['s_grd_b'], ctx['s_grd_c'], ctx['s_grd_l1'], ctx['s_grd_l2'] = self.grd_scorer(x, self.label)
        ctx['s_grd2_a'], ctx['s_grd2_b'], ctx['s_grd2_c'], ctx['s_grd2_l1'], ctx['s_grd2_l2'] = self.grd_scorer2(x, self.label)


class MF_SecondOrderSDPScorer(nn.Module):
    def __init__(self, conf, fields, input_dim):
        super(MF_SecondOrderSDPScorer, self).__init__()
        self.conf = conf

        self.rel_scorer = BiaffineScorer(n_in=input_dim, n_out=conf.n_mlp_rel, bias_x=True, bias_y=True,
                                         dropout=conf.mlp_dropout, n_out_label=fields.get_vocab_size("rel"),
                                         scaling=conf.scaling)

        self.arc_scorer = BiaffineScorer(n_in=input_dim, n_out=conf.n_mlp_arc, bias_x=True, bias_y=True,
                                         dropout=conf.mlp_dropout, n_out_label=fields.get_vocab_size("rel"),
                                         scaling=conf.scaling)

        self.sib_scorer = DecomposedTriAffineScorer(n_in=input_dim, n_out=conf.n_mlp_rel,
                                                    dropout=conf.mlp_dropout, rank=conf.rank, scaling=self.conf.scaling)

        # self.cop_scorer = DecomposedTriAffineScorer(n_in=input_dim, n_out=conf.n_mlp_rel,
        #                                             dropout=conf.mlp_dropout, rank=conf.rank, scaling=self.conf.scaling)

        self.grd_scorer = DecomposedTriAffineScorer(n_in=input_dim, n_out=conf.n_mlp_rel,
                                                    dropout=conf.mlp_dropout, rank=conf.rank, scaling=self.conf.scaling)

        self.grd_scorer2 = DecomposedTriAffineScorer(n_in=input_dim, n_out=conf.n_mlp_rel,
                                                     dropout=conf.mlp_dropout, rank=conf.rank,
                                                     scaling=self.conf.scaling)

    def forward(self, ctx):
        x = ctx['encoded_emb'][:, :-1]
        ctx['s_rel'] =  self.rel_scorer(x)
        ctx['s_arc'] = self.arc_scorer(x)
        ctx['s_sib_a'], ctx['s_sib_b'], ctx['s_sib_c'] = self.sib_scorer(x, self.label)
        # ctx['s_cop_a'], ctx['s_cop_b'], ctx['s_cop_c'] = self.cop_scorer(x, self.label)
        ctx['s_grd_a'], ctx['s_grd_b'], ctx['s_grd_c'] = self.grd_scorer(x, self.label)
        ctx['s_grd2_a'], ctx['s_grd2_b'], ctx['s_grd2_c'] = self.grd_scorer2(x, self.label)

