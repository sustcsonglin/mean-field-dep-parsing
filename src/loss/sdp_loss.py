from supar.utils.common import *
import torch
import torch.nn.functional as F
import torch.nn as nn
from opt_einsum import contract
bce = nn.BCEWithLogitsLoss()
cross_entropy  = nn.CrossEntropyLoss()



class CPD_MFLoss():
    def __init__(self, conf):
        self.conf = conf

    def loss(self, ctx):
        seq_len = ctx['seq_len']
        batch_size = seq_len.shape[0]
        max_len = seq_len.max() + 1
        ### label loss.
        gold_rel = ctx['rel']

        marginal_logits = CPDLabeledMFI(ctx, max_iter=self.conf.max_iter)
        sent_masks = torch.arange(seq_len.max()+1, device=seq_len.device)[None, :] <= seq_len[:, None]
        sent_masks = sent_masks.unsqueeze(1) & sent_masks.unsqueeze(2)
        # root cannot be child.
        # sent_masks[:, :, 0] = False
        # self loop is not allowed.
        # sent_masks[:, torch.arange(seq_len.max()+1), torch.arange(seq_len.max()+1)] = False
        label_masks = seq_len.new_zeros(batch_size, max_len, max_len)
        label_masks[gold_rel[:, 0], gold_rel[:, 1], gold_rel[:, 2]] = torch.as_tensor(gold_rel[:, -1], device=seq_len.device, dtype=torch.long)
        label_loss = cross_entropy(marginal_logits[sent_masks], label_masks[sent_masks])
        return label_loss

    def decode(self, ctx):
        marginal_logit = CPDLabeledMFI(ctx, max_iter=10)

        s_rel = ctx['s_rel']
        gold_rel = ctx['rel']
        seq_len = ctx['seq_len']
        max_len = seq_len.max() + 1
        masks = torch.arange(max_len, device=seq_len.device)[None, :] <= seq_len[:, None]
        masks = masks.unsqueeze(1) & masks.unsqueeze(2)
        masks[:, torch.arange(max_len), torch.arange(max_len)] = False
        masks[:, :, 0] = False
        pred = marginal_logit.argmax(-1)
        # The 0th label <pad> is used as <NULL> label.
        pred.masked_fill_(pred==0, -1)
        # Invalid positions.
        pred.masked_fill_(~masks, -1)
        gold = marginal_logit.new_zeros(s_rel.shape[0], s_rel.shape[1], s_rel.shape[2], dtype=torch.long).fill_(-1)
        gold[gold_rel[:, 0], gold_rel[:, 1], gold_rel[:, 2]] = torch.as_tensor(gold_rel[:, -1], device=s_rel.device, dtype=torch.long)
        ctx['gold'] = gold
        ctx['pred'] = pred



def CPDLabeledMFI(ctx, max_iter, type='softmax', alpha=1.0):
    seq_len = ctx['seq_len']
    max_len = int(seq_len.max() + 1)
    mask = torch.arange(max_len, device=seq_len.device)[None, :]  <= seq_len[:, None]
    mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)

    # root cannot be child.
    # mask[:, :, 0] = False
    # self-loop is not allowed.
    # mask[:, torch.arange(max_len), torch.arange(max_len)] = False

    s_sib_a, s_sib_b, s_sib_c, s_sib_l1, s_sib_l2 =  ctx['s_sib_a'], ctx['s_sib_b'], ctx['s_sib_c'], ctx['s_sib_l1'], ctx['s_sib_l2']
    s_cop_a, s_cop_b, s_cop_c, s_cop_l1, s_cop_l2 = ctx['s_cop_a'], ctx['s_cop_b'], ctx['s_cop_c'], ctx['s_cop_l1'], ctx['s_cop_l2']
    s_grd_a, s_grd_b, s_grd_c, s_grd_l1, s_grd_l2 = ctx['s_grd_a'], ctx['s_grd_b'], ctx['s_grd_c'], ctx['s_grd_l1'], ctx['s_grd_l2']
    s_grd2_a, s_grd2_b, s_grd2_c, s_grd2_l1, s_grd2_l2 = ctx['s_grd2_a'], ctx['s_grd2_b'], ctx['s_grd2_c'], ctx['s_grd2_l1'], ctx['s_grd2_l2']
    s_rel = ctx['s_rel']
    q = s_rel.clone()
    mask = mask.unsqueeze(-1).expand(*q.shape).float()

    # root cannot be siblings.
    # s_sib_a[:, 0] = 0
    # s_sib_b[:, 0] = 0
    # s_sib_c[:, 0] = 0

    # # root cannot be coparent.
    # s_cop_a[:, 0] = 0
    # s_cop_b[:, 0] = 0
    # s_cop_c[:, 0] = 0
    # #
    # # root cannot be grandson. root can be grandparent.
    # s_grd_b[:, 0] = 0
    # s_grd_c[:, 0] = 0
    # s_grd2_a[:, 0] = 0
    # s_grd2_b[:, 0] = 0

    for i in range(max_iter):
        q = q.softmax(-1)
        q = q*mask
        s_rel_new = s_rel
        # # \sum_k q(b,i,k,m)s^sib(b,i, j, k, l, m)->q(b, i, j, l), \sum_k q(b, k, j, l) s^cop(b, i, j, k, l, m)-> q(b, i, j, l)
        # # \sum_k q(b,j,k,m)s^grd(b,i, j, k, l, m)->q(b, i, j, l), \sum_k q(b, k, i, m) s^grd2(b, i, j, k, l, m)-> q(b, i, j, l)
        # # -------------sib score -------------
        tmp1 = contract('nabl, nar, nbr, lr -> nar', q, s_sib_a, s_sib_c, s_sib_l1, backend='torch')
        s_rel_new = s_rel_new + contract('nar, nbr, lr -> nabl', tmp1, s_sib_b, s_sib_l2, backend='torch')

        # s_rel_new = s_rel_new - contract('nabl, nar, nbr, nbr, lr -> nabr',
        #                         q, s_sib_a, s_sib_b, s_sib_c, s_sib_l1, backend='torch') @ s_sib_l2.t()
        #
        # -------------sib score -------------
        # -------------cop score -------------
        tmp1 = contract('nabl, nar, nbr, lr -> nbr', q, s_cop_c, s_cop_b, s_cop_l1, backend='torch')
        s_rel_new = s_rel_new + contract('nar, nbr, lr -> nabl', s_cop_a, tmp1, s_cop_l2, backend='torch')

        # s_rel_new = s_rel_new - contract('nabl, nar, nar, nbr, lr -> nabr',
        #                       q, s_cop_a, s_cop_c, s_cop_b, s_cop_l1, backend='torch') @ s_cop_l2.t()

        # --------------cop score ----
        # # grand score 1
        tmp1 = contract('nabl, nar, nbr, lr -> nar', q, s_grd_b, s_grd_c, s_grd_l1, backend='torch')
        s_rel_new = s_rel_new + contract('nar, nbr, lr -> nabl', s_grd_a, tmp1, s_grd_l2, backend='torch')

        # s_rel_new = s_rel_new - contract('nabl, nar, nbr, nbr, lr -> nbar',
        #                       q, s_grd_b, s_grd_c, s_grd_a, s_grd_l1, backend='torch') @ s_grd_l2.t()

        # grand score2
        tmp1 = contract('nabl, nar, nbr, lr -> nbr', q, s_grd2_c, s_grd2_a, s_grd2_l1, backend='torch')
        s_rel_new = s_rel_new + contract('nar, nbr, lr -> nabl', tmp1, s_grd2_b, s_grd2_l2, backend='torch')

        # q =  s_rel_new - contract('nkil, nkr, nkr, nir, lr-> nikr',
        #                       q, s_grd2_c, s_grd2_b, s_grd2_a, s_grd2_l1, backend='torch') @ s_grd2_l2.t()

        q = s_rel_new

    return q






