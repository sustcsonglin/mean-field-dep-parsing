
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

