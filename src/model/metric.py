
import logging
import sys
from asyncio import Queue
from collections import Counter
from queue import Empty

import nltk
import subprocess
import torch
from pytorch_lightning.metrics import Metric
from threading import Thread
import regex

from supar.utils.transform import Tree
import tempfile

log = logging.getLogger(__name__)
import os



class SDPMetric(Metric):
    def __init__(self, cfg, fields):
        super().__init__()
        self.fields = fields
        self.cfg = cfg
        self.vocab = self.fields.get_vocab('rel')

        self.add_state("tp", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("utp", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("pred", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("gold", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_pred", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_gold", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_ucm", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_lcm", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0.), dist_reduce_fx="sum")

        self.eps = 1e-12

        if self.cfg.write_result_to_file:
            self.add_state("outputs", default=[])
            self.prefix = "dep"

    def update(self, ctx):
        preds = ctx['pred']
        golds = ctx['gold']

        pred_mask = preds.ge(0)
        gold_mask = golds.ge(0)
        span_mask = pred_mask & gold_mask
        self.pred += pred_mask.sum().item()
        self.gold += gold_mask.sum().item()
        self.tp += (preds.eq(golds) & span_mask).sum().item()
        self.utp += span_mask.sum().item()

        # if self.cfg.write_result_to_file:
        #     outputs = {}
        #     outputs['arc_preds'] = arc_preds.detach().cpu().numpy()
        #     outputs['rel_preds'] = rel_preds.detach().cpu().numpy()
        #     outputs['raw_word'] = ctx['raw_word']
        #     outputs['id'] = ctx['word_id']
        #     self.outputs.append(outputs)

    def compute(self, test=True, epoch_num=-1):
        super(SDPMetric, self).compute()
        if self.cfg.write_result_to_file and (epoch_num > 0 or test):
            self._write_result_to_file(test=test)
        return self.result

    @property
    def result(self):
        return {
                'up': self.up,
                'ur': self.ur,
                'uf': self.uf,
                'p': self.p,
                'r': self.r,
                'f': self.f,
            'score': self.f
                }

    @property
    def score(self):
        return self.f.item()

    @property
    def up(self):
        return (self.utp / (self.pred + self.eps)).item()

    @property
    def ur(self):
        return (self.utp / (self.gold + self.eps)).item()


    @property
    def uf(self):
        return (2 * self.utp / (self.pred + self.gold + self.eps)).item()

    @property
    def p(self):
        return (self.tp / (self.pred + self.eps)).item()

    @property
    def r(self):
        return (self.tp / (self.gold + self.eps)).item()

    @property
    def f(self):
        return (2 * self.tp / (self.pred + self.gold + self.eps)).item()




class SDPMetric2(Metric):
    def __init__(self, cfg, fields):
        super().__init__()
        self.fields = fields
        self.cfg = cfg
        self.vocab = self.fields.get_vocab('rel')

        self.add_state("tp", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("utp", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("pred", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("gold", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_pred", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_gold", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_ucm", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_lcm", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0.), dist_reduce_fx="sum")

        self.eps = 1e-12

        if self.cfg.write_result_to_file:
            self.add_state("outputs", default=[])
            self.prefix = "sdp"

    def update(self, ctx):
        preds = ctx['pred']
        golds = ctx['gold']
        pred_mask = preds.ge(0)
        gold_mask = golds.ge(0)
        span_mask = pred_mask & gold_mask
        self.pred += pred_mask.sum().item()
        self.gold += gold_mask.sum().item()
        self.tp += (preds.eq(golds) & span_mask).sum().item()
        self.utp += span_mask.sum().item()

        if self.cfg.write_result_to_file:
            batch_size = preds.shape[0]
            outputs = {}
            seq_len = ctx['seq_len'].cpu().numpy()
            pred_arc = pred_mask.nonzero().tolist()
            preds = preds.cpu().numpy()
            pred_arc_label = [[[] for _ in range(seq_len[i])] for i in range(batch_size)]
            for i in range(len(pred_arc)):
                pred_arc_label[pred_arc[i][0]][pred_arc[i][2]-1].append([pred_arc[i][1], self.vocab[preds[pred_arc[i][0], pred_arc[i][1], pred_arc[i][2]]]])
            outputs['pred_arc_label'] = pred_arc_label
            outputs['raw_word'] = ctx['raw_word']
            outputs['id'] = ctx['word_id']
            self.outputs.append(outputs)

    def compute(self, mode):
        super(SDPMetric2, self).compute()
        if self.cfg.write_result_to_file:
            self._write_result_to_file(mode=mode)
        return self.result

    @property
    def result(self):
        return {
                'up': self.up,
                'ur': self.ur,
                'uf': self.uf,
                'p': self.p,
                'r': self.r,
                'f': self.f,
            'score': self.f
                }

    @property
    def score(self):
        return self.f.item()

    @property
    def up(self):
        return (self.utp / (self.pred + self.eps)).item()

    @property
    def ur(self):
        return (self.utp / (self.gold + self.eps)).item()

    @property
    def uf(self):
        return (2 * self.utp / (self.pred + self.gold + self.eps)).item()

    @property
    def p(self):
        return (self.tp / (self.pred + self.eps)).item()

    @property
    def r(self):
        return (self.tp / (self.gold + self.eps)).item()

    @property
    def f(self):
        return (2 * self.tp / (self.pred + self.gold + self.eps)).item()

    def _write_result_to_file(self, mode):
        outputs = self.outputs
        ids = [output['id'] for output in outputs]
        raw_word = [output['raw_word'] for output in outputs]
        preds = [output['pred_arc_label'] for output in outputs]
        total_len =  sum(batch.shape[0] for batch in ids)
        final_results = [[] for _ in range(total_len)]
        for batch in zip(ids, raw_word, preds):
            batch_ids, batch_word, batch_arc = batch
            for i in range(batch_ids.shape[0]):
                length = len(batch_word[i])
                for j in range(length):
                    final_results[batch_ids[i]].append([batch_word[i][j], batch_arc[i][j]])

        with open(f"{self.prefix}_output_{mode}.txt", 'w', encoding='utf8') as f:
            for result in final_results:
                for line_id, (word, arc_rel) in enumerate(result, start=1):
                    if len(arc_rel) == 0:
                        f.write('\t'.join(
                            [str(line_id), word, '-', '-', '-', '-',
                             '-',
                             '-', '-', '-', '-']))
                    else:
                        f.write('\t'.join(
                            [str(line_id), word, '-', '-', '-', '-',
                             '-',
                             '-', '-', '|'.join(f"{head}:{label}" for head, label in arc_rel), '-']))

                    f.write('\n')
                f.write('\n')

        self.outputs.clear()



class AttachmentMetric(Metric):
    def __init__(self, cfg, fields):
        super().__init__()
        self.fields = fields
        self.cfg = cfg
        self.vocab = self.fields.get_vocab('rel')

        self.add_state("correct_arcs", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("correct_rels", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_ucm", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_lcm", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0.), dist_reduce_fx="sum")

        self.eps = 1e-12

        if self.cfg.write_result_to_file:
            self.add_state("outputs", default=[])
            self.prefix = "dep"


    def update(self, info):

        arc_preds = info['arc_pred']
        arc_gold = info['arc_gold']
        mask_dep = info['mask_dep']
        rel_preds = info['rel_pred']
        rel_gold = info['rel_gold']

        arc_mask = arc_preds.eq(arc_gold) & mask_dep
        rel_mask = rel_preds.eq(rel_gold) & arc_mask
        arc_mask_seq, rel_mask_seq = arc_mask[mask_dep], rel_mask[mask_dep]
        self.n += len(mask_dep)
        lens = mask_dep.sum(1)
        self.n_ucm += arc_mask.sum(1).eq(lens).sum()
        self.n_lcm += rel_mask.sum(1).eq(lens).sum()
        self.total += len(arc_mask_seq)
        self.correct_arcs += arc_mask_seq.sum()
        self.correct_rels += rel_mask_seq.sum()

        if self.cfg.write_result_to_file:
            outputs = {}
            outputs['arc_preds'] = arc_preds.detach().cpu().numpy()
            outputs['rel_preds'] = rel_preds.detach().cpu().numpy()
            outputs['raw_word'] = info['raw_word']
            outputs['id'] = info['word_id']
            self.outputs.append(outputs)

    def compute(self, test=True, epoch_num=-1):
        super(AttachmentMetric, self).compute()
        if self.cfg.write_result_to_file and (epoch_num > 0 or test):
            self._write_result_to_file(test=test)
        return self.result



    def _write_result_to_file(self, test=False):
        mode = 'test' if test else 'valid'
        outputs = self.outputs
        ids = [output['id'] for output in outputs]
        raw_word = [output['raw_word'] for output in outputs]
        arc_preds = [output['arc_preds'] for output in outputs]
        rel_preds = [output['rel_preds'] for output in outputs]

        total_len =  sum(batch.shape[0] for batch in ids)

        final_results = [None for _ in range(total_len)]

        for batch in zip(ids, raw_word, arc_preds, rel_preds):
            batch_ids, batch_word, batch_arc, batch_rel = batch

            for i in range(batch_ids.shape[0]):
                length = len(batch_word[i])
                # recall that the first token is the imaginary root;
                final_results[batch_ids[i]] = [batch_word[i],
                                               batch_arc[i][1:length+1],
                                               self.vocab[batch_rel[i][1:length+1]]
                                               ]

        with open(f"{self.prefix}_output_{mode}.txt", 'w', encoding='utf8') as f:
            for (words, arcs, rels) in final_results:
                for line_id, (word, arc, rel) in enumerate(zip(words, arcs, rels), start=1):
                    # 1=word, 6=arc, 7=rel
                    f.write('\t'.join(
                        [str(line_id), word, '-', '-', '-', '-',
                         str(arc),
                         str(rel), '-', '-', '-']))
                    f.write('\n')
                f.write('\n')



    @property
    def result(self):
        return {'d_ucm': (self.n_ucm / (self.n + self.eps)).item(),
                'd_lcm': (self.n_lcm / (self.n + self.eps)).item(),
                'uas': (self.correct_arcs / (self.total + self.eps)).item(),
                'las': (self.correct_rels / (self.total + self.eps)).item(),
                'score': self.las
                }


    def score(self):
        return self.las

    @property
    def ucm(self):
        return (self.n_ucm / (self.n + self.eps)).item()


    @property
    def lcm(self):
        return (self.n_lcm / (self.n + self.eps)).item()

    @property
    def uas(self):
        return (self.correct_arcs / (self.total + self.eps)).item()

    @property
    def las(self):
        return (self.correct_rels / (self.total + self.eps)).item()



class PseudoProjDepExternalMetric(Metric):
    def __init__(self, cfg, fields):
        super().__init__()
        self.fields = fields
        self.cfg = cfg
        self.add_state("outputs", default=[])
        self.add_state("n", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("las", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state("las_conll18", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state("uas", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.prefix = "ud"
        self.ignore_punct = (not cfg.eval_punct)

    def update(self, ctx):
        arc_preds = ctx['arc_pred']
        rel_preds = ctx['rel_pred']
        raw_word = ctx['raw_word']
        batch_id = ctx['id']
        outputs = {}
        outputs['arc_preds'] = arc_preds.detach().cpu().numpy()
        outputs['rel_preds'] = rel_preds.detach().cpu().numpy()
        outputs['raw_word'] = raw_word
        outputs['id'] = batch_id
        self.outputs.append(outputs)


    @property
    def result(self):
        return {'uas': self.uas,
                'las': self.las,
                'las_conll18': self.las_conll18,
                'score': self.las_conll18}

    def compute(self, test=False, epoch_num=-1):
        # 同步outputs
        super().compute()
        if epoch_num > 0:
            self._write_result_to_file(test=test)
            mode = 'test' if test else 'valid'
            # mode2 = 'test' if test else 'dev'
            gold_file = self.fields.conf.test_dep if test else self.fields.conf.dev_dep
            src_file = os.getcwd() + f"/{self.prefix}_output_{mode}.txt"
            # tgt_file = os.getcwd() + f"/{self.prefix}.output_transformed_{mode}.txt"
            uas, las, las_conll18 = self._eval(self._load(src_file), self._load(gold_file))
            self.las += las
            self.uas += uas
            self.las_conll18 += las_conll18
        return self.result

    def _load(self, fname):
        sents = []
        with open(fname, encoding='utf8') as f:
            sent = []
            for line in f.readlines():
                if line[0] == '#':
                    continue
                line = line.strip().split('\t')
                if len(line) > 7:
                    word_id, word, arc, rel = line[0], line[1], line[6], line[7]
                    if word_id.isdigit():
                        sent.append((word, arc, rel))
                elif len(sent):
                    sents.append(sent)
                    sent = []
            if len(sent):
                sents.append(sent)
        return sents

    def _eval(self, predicts, golds):
        total = 0
        u_correct = 0
        l_correct = 0
        l_correct_conll18 = 0

        try:
            assert len(predicts) == len(golds), "total num mismatch"
            for s_idx, (predict, gold) in enumerate(zip(predicts, golds)):
                assert len(predict) == len(gold), f"{s_idx} instance mismatch"
                for w_idx, ((_, p_a, p_r), (g_w, g_a, g_r)) in enumerate(zip(predict, gold)):
                    # assert (regex.match(r'\p{P}+$', g_w) is not None) == all(unicodedata.category(char).startswith('P') for char in g_w)
                    if self.ignore_punct and regex.match(r'\p{P}+$', g_w):
                        continue
                    total += 1
                    if p_a == g_a:
                        u_correct += 1
                        if p_r == g_r:
                            l_correct += 1
                        if p_r.split(":")[0] == g_r.split(":")[0]:
                            l_correct_conll18 += 1
            return u_correct / total, l_correct / total, l_correct_conll18 / total
        except:
            return 0., 0.

    def _write_result_to_file(self, test=False):
        mode = 'test' if test else 'valid'
        outputs = self.outputs
        ids = [output['id'] for output in outputs]
        raw_word = [output['raw_word'] for output in outputs]
        arc_preds = [output['arc_preds'] for output in outputs]
        rel_preds = [output['rel_preds'] for output in outputs]

        total_len = sum(batch.shape[0] for batch in ids)

        final_results = [None for _ in range(total_len)]

        self.vocab = self.fields.get_vocab('rel')

        for batch in zip(ids, raw_word, arc_preds, rel_preds):
            batch_ids, batch_word, batch_arc, batch_rel = batch
            for i in range(batch_ids.shape[0]):
                length = len(batch_word[i])
                # recall that the first token is the imaginary root;
                final_results[batch_ids[i]] = [batch_word[i],
                                               batch_arc[i][1:length + 1],
                                               self.vocab[batch_rel[i][1:length + 1]]
                                               ]

        with open(f"{self.prefix}_output_{mode}.txt", 'w', encoding='utf8') as f:
            for (words, arcs, rels) in final_results:
                for line_id, (word, arc, rel) in enumerate(zip(words, arcs, rels), start=1):
                    # 1=word, 6=arc, 7=rel
                    f.write('\t'.join(
                        [str(line_id), word, '-', '-', '-', '-',
                         str(arc),
                         str(rel), '-', '-']))
                    f.write('\n')
                f.write('\n')

        # s
        # elf.outputs.clear()

