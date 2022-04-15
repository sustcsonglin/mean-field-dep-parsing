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
        self.inference = CPDLabeledMFI

    def loss(self, ctx):
        seq_len = ctx['seq_len']

        gold_head = torch.as_tensor(ctx['head'], device=seq_len.device, dtype=seq_len.dtype)
        gold_rel = torch.as_tensor(ctx['rel'], device=seq_len.device, dtype=seq_len.dtype)


        marginal_logits = self.inference(ctx, max_iter=self.conf.max_iter)
        # max-margin loss. Projective parsing only.
        if self.conf.loss_type == 'mm':

            with torch.no_grad():
                s_rel = marginal_logits.clone()
                s_rel[gold_head[:, 0], gold_head[:, 2], gold_head[:, 1], gold_rel[:, -1]] -= 1
                s_arc, s_rel_idx = s_rel.max(-1)
                ctx['s_arc'] = s_arc.transpose(1, 2)
                arc = eisner(ctx, max_margin=True)
                rel = s_rel_idx[arc[:, 0], arc[:, 2], arc[:, 1]]

            mm_score = marginal_logits[arc[:, 0], arc[:, 2], arc[:, 1], rel].sum()
            gold_score = marginal_logits[gold_head[:, 0], gold_head[:, 2], gold_head[:, 1], gold_rel[:, -1]].sum()
            loss = (mm_score - gold_score) / seq_len.sum()

        elif self.conf.loss_type == 'head_selection':
            raise NotImplementedError

        return loss

    def decode(self, ctx):
        s_rel = CPDLabeledMFI(ctx, max_iter=self.conf.max_iter)
        # mask out the <null> label.
        s_arc, s_rel_idx = s_rel.max(-1)
        ctx['s_arc'] = s_arc.transpose(1, 2)
        seq_len = ctx['seq_len']

        if self.conf.decode_type == 'eisner':
            eisner(ctx, decode=True)
            arc_preds = ctx['arc_pred']
            rel_pred = s_rel_idx.gather(-2, arc_preds.unsqueeze(-2)).squeeze(-2)
        else:
            # mst
            raise NotImplementedError

        # --------- POST-Process for evaluation ----------------------
        arc_golds, rel_golds = ctx['head'], ctx['rel']
        arc_golds = torch.as_tensor(arc_golds, device=seq_len.device, dtype=seq_len.dtype)
        rel_golds = torch.as_tensor(rel_golds, device=seq_len.device, dtype=seq_len.dtype)
        arc_gold = arc_golds.new_zeros(*arc_preds.shape).fill_(-1)
        arc_gold[arc_golds[:, 0], arc_golds[:, 1]] = arc_golds[:, 2]
        rel_gold = arc_preds.new_zeros(*arc_preds.shape).fill_(-1)
        rel_gold[rel_golds[:, 0], rel_golds[:, 1]] = rel_golds[:, 2]
        mask_dep = arc_gold.ne(-1)
        # ignore punct.
        if 'is_punct' in ctx:
            mask_punct = ctx['is_punct'].nonzero()
            mask_dep[mask_punct[:, 0], mask_punct[:, 1] + 1] = False
        ctx['arc_gold'] = arc_gold
        ctx['rel_gold'] = rel_gold
        ctx['mask_dep'] = mask_dep
        ctx['arc_pred'] = arc_preds
        ctx['rel_pred'] = rel_pred


class MFLoss():
    def __init__(self, conf):
        self.conf = conf
        self.inference = MFI

    def loss(self, ctx):
        seq_len = ctx['seq_len']

        ### label loss.
        gold_head = torch.as_tensor(ctx['head'], device=seq_len.device, dtype=seq_len.dtype)
        gold_rel = torch.as_tensor(ctx['rel'], device=seq_len.device, dtype=seq_len.dtype)
        s_rel = ctx['s_rel']
        label_loss = F.cross_entropy(s_rel[gold_head[:, 0], gold_head[:, 2], gold_head[:, 1]], torch.as_tensor(gold_rel[:, -1], device=s_rel.device, dtype=torch.long))

        marginal_logits = self.inference(ctx, max_iter=self.conf.max_iter)
        # max-margin loss. Projective parsing only.
        if self.conf.loss_type == 'mm':
            with torch.no_grad():
                s_arc = marginal_logits.clone()
                s_arc[gold_head[:, 0], gold_head[:, 2], gold_head[:, 1]] -= 1
                ctx['s_arc'] = s_arc.transpose(1, 2)
                arc = eisner(ctx, max_margin=True)
            mm_score = marginal_logits[arc[:, 0], arc[:, 2], arc[:, 1]].sum()
            gold_score = marginal_logits[gold_head[:, 0], gold_head[:, 2], gold_head[:, 1]].sum()
            tree_loss = (mm_score - gold_score) / seq_len.sum()

        elif self.conf.loss_type == 'head_selection':
            raise NotImplementedError

        return label_loss + tree_loss

    def decode(self, ctx):
        s_arc = CPDLabeledMFI(ctx, max_iter=self.conf.max_iter)
        seq_len = ctx['seq_len']

        if self.conf.decode_type == 'eisner':
            ctx['s_arc'] = s_arc.transpose(1, 2)
            eisner(ctx, decode=True)
            arc_preds = ctx['arc_pred']
        else:
            # mst
            raise NotImplementedError

        s_rel = ctx['s_rel']
        rel_preds = s_rel.argmax(-1).gather(-2, arc_preds.unsqueeze(-2)).squeeze(-2)

        # --------- POST-Process for evaluation ----------------------
        arc_golds, rel_golds = ctx['head'], ctx['rel']
        arc_golds = torch.as_tensor(arc_golds, device=seq_len.device, dtype=seq_len.dtype)
        rel_golds = torch.as_tensor(rel_golds, device=seq_len.device, dtype=seq_len.dtype)
        arc_gold = arc_golds.new_zeros(*arc_preds.shape).fill_(-1)
        arc_gold[arc_golds[:, 0], arc_golds[:, 1]] = arc_golds[:, 2]
        rel_gold = arc_preds.new_zeros(*arc_preds.shape).fill_(-1)
        rel_gold[rel_golds[:, 0], rel_golds[:, 1]] = rel_golds[:, 2]
        mask_dep = arc_gold.ne(-1)
        # ignore punct.
        if 'is_punct' in ctx:
            mask_punct = ctx['is_punct'].nonzero()
            mask_dep[mask_punct[:, 0], mask_punct[:, 1] + 1] = False
        ctx['arc_gold'] = arc_gold
        ctx['rel_gold'] = rel_gold
        ctx['mask_dep'] = mask_dep
        ctx['arc_pred'] = arc_preds
        ctx['rel_pred'] = rel_preds



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
    # s_cop_a, s_cop_b, s_cop_c, s_cop_l1, s_cop_l2 = ctx['s_cop_a'], ctx['s_cop_b'], ctx['s_cop_c'], ctx['s_cop_l1'], ctx['s_cop_l2']
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
        # tmp1 = contract('nabl, nar, nbr, lr -> nbr', q, s_cop_c, s_cop_b, s_cop_l1, backend='torch')
        # s_rel_new = s_rel_new + contract('nar, nbr, lr -> nabl', s_cop_a, tmp1, s_cop_l2, backend='torch')

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


def MFI(ctx, max_iter):
    seq_len = ctx['seq_len']
    max_len = seq_len.max() + 1
    mask = torch.arange(max_len, device=seq_len.device)[None, :]  <= seq_len[:, None]
    mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)

    s_sib_a, s_sib_b, s_sib_c = ctx['s_sib_a'], ctx['s_sib_b'], ctx['s_sib_c']
    s_grd1_a, s_grd1_b, s_grd1_c = ctx['s_grd1_a'], ctx['s_grd1_b'], ctx['s_grd1_c']
    s_grd2_a, s_grd2_b, s_grd2_c = ctx['s_grd2_a'], ctx['s_grd2_b'], ctx['s_grd2_c']
    s_arc = ctx['s_arc']
    q = s_arc

    # \sum_k q(b,i,k,m)s^sib(b,i, j, k, l, m)->q(b, i, j, l), \sum_k q(b, k, j, l) s^cop(b, i, j, k, l, m)-> q(b, i, j, l)
    # \sum_k q(b,j,k,m)s^grd(b,i, j, k, l, m)->q(b, i, j, l), \sum_k q(b, k, i, m) s^grd2(b, i, j, k, l, m)-> q(b, i, j, l)
    for _ in range(max_iter):
        q = q.sigmoid()
        q = q * mask
        q_new = s_arc + torch.einsum('nar, nbr, ncr, nac -> nab', s_sib_a, s_sib_b, s_sib_c, q) \
                + torch.einsum('nar, nbr, ncr, nbc -> nab', s_grd1_a, s_grd1_b, s_grd1_c, q) \
                + torch.einsum('nar, nbr, ncr, nab -> nbc', s_grd2_a, s_grd2_b, s_grd2_c, q)
        q = q_new

    return q

@torch.enable_grad()
def eisner(ctx, decode=False, max_margin=False):

    scores = ctx['s_arc']
    lens = ctx['seq_len']

    if decode or max_margin:
        scores_origin = scores.detach().clone().requires_grad_(True)
        assert scores_origin.requires_grad
    else:
        scores_origin = scores
    # the end position of each sentence in a batch
    batch_size, seq_len, _ = scores.shape
    # [seq_len, seq_len, batch_size]
    scores = scores_origin.permute(2, 1, 0)
    s_i = torch.full_like(scores, float('-inf'))
    s_c = torch.full_like(scores, float('-inf'))
    s_c.diagonal().fill_(0)
    # set the scores of arcs excluded by cands to -inf
    viterbi = decode or max_margin

    for w in range(1, seq_len):
        # n denotes the number of spans to iterate,
        # from span (0, w) to span (n, n+w) given width w
        n = seq_len - w
        # ilr = C(i->r) + C(j->r+1)
        # [n, w, batch_size]
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        if ilr.requires_grad:
            ilr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))

        if viterbi:
            il = ir = ilr.permute(2, 0, 1).max(-1)[0]
        else:
            il = ir = ilr.permute(2, 0, 1).logsumexp(-1)

        # I(j->i) = logsumexp(C(i->r) + C(j->r+1)) + s(j->i), i <= r < j
        # fill the w-th diagonal of the lower triangular part of s_i
        # with I(j->i) of n spans
        s_i.diagonal(-w).copy_(il + scores.diagonal(-w))
        # I(i->j) = logsumexp(C(i->r) + C(j->r+1)) + s(i->j), i <= r < j
        # fill the w-th diagonal of the upper triangular part of s_i
        # with I(i->j) of n spans
        s_i.diagonal(w).copy_(ir + scores.diagonal(w))

        # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        if cl.requires_grad:
            cl.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))

        if viterbi:
            s_c.diagonal(-w).copy_(cl.permute(2, 0, 1).max(-1)[0])
        else:
            s_c.diagonal(-w).copy_(cl.permute(2, 0, 1).logsumexp(-1))

        # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j

        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        if cr.requires_grad:
            cr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))

        if viterbi:
            s_c.diagonal(w).copy_(cr.permute(2, 0, 1).max(-1)[0])
        else:
            s_c.diagonal(w).copy_(cr.permute(2, 0, 1).logsumexp(-1))

        # disable multi words to modify the root
        s_c[0, w][lens.ne(w)] = float('-inf')

    logZ = s_c[0].gather(0, lens.unsqueeze(0))

    if not decode and not max_margin:
        return logZ

    logZ.sum().backward()

    if decode:
        dep = scores_origin.grad
        predicted_arc = dep.new_zeros(dep.shape[0], dep.shape[1]).long()
        arc = dep.nonzero()
        predicted_arc[arc[:, 0], arc[:, 1]] = arc[:, 2]
        ctx['arc_pred'] = predicted_arc
        return
    else:
        return  scores_origin.grad.nonzero()


def stripe(x, n, w, offset=(0, 0), dim=1):
    # r"""
    # Returns a diagonal stripe of the tensor.
    #
    # Args:
    #     x (~torch.Tensor): the input tensor with 2 or more dims.
    #     n (int): the length of the stripe.
    #     w (int): the width of the stripe.
    #     offset (tuple): the offset of the first two dims.
    #     dim (int): 1 if returns a horizontal stripe; 0 otherwise.
    #
    # Returns:
    #     a diagonal stripe of the tensor.
    # Examples:
    #     >>> x = torch.arange(25).view(5, 5)
    #     >>> x
    #     tensor([[ 0,  1,  2,  3,  4],
    #             [ 5,  6,  7,  8,  9],
    #             [10, 11, 12, 13, 14],
    #             [15, 16, 17, 18, 19],
    #             [20, 21, 22, 23, 24]])
    #     >>> stripe(x, 2, 3)
    #     tensor([[0, 1, 2],
    #             [6, 7, 8]])
    #     >>> stripe(x, 2, 3, (1, 1))
    #     tensor([[ 6,  7,  8],
    #             [12, 13, 14]])
    #     >>> stripe(x, 2, 3, (1, 1), 0)
    #     tensor([[ 6, 11, 16],
    #             [12, 17, 22]])
    # """
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[0, 0].numel()
    stride[0] = (seq_len + 1) * numel
    stride[1] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(n, w, *x.shape[2:]),
                        stride=stride,
                        storage_offset=(offset[0]*seq_len+offset[1])*numel)
