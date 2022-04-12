import pytorch_lightning as pl
import os
from supar.utils.common import *
import pickle
from .dm_util.fields import SubwordField, Field, SpanField
from .dm_util.padder import *
import logging
log = logging.getLogger(__name__)
from .base import DataModuleBase
from fastNLP.io.loader.conll import ConllLoader
from .dm_util.util import *
import tempfile
from supar.utils.transform import CoNLL
from .dm_util.datamodule_util import get_sampler
from fastNLP.core.batch import DataSetIter


class SDPData(DataModuleBase):
    def __init__(self, conf):
        super(SDPData, self).__init__(conf)

    def get_inputs(self):

        inputs = ['seq_len', 'raw_word', 'id']

        if self.conf.use_pos:
            inputs.append('pos')

        if self.conf.use_lemma:
            inputs.append('lemma')

        return inputs

    def get_targets(self):
        return ['rel']

    def build_datasets(self):
        datasets = {}
        conf = self.conf
        datasets['train'] = self._load(dep_file=conf.train, mode='train')
        datasets['dev'] = self._load(dep_file=conf.dev, mode='dev')
        datasets['test'] = self._load(dep_file=conf.test_id, mode='test')
        datasets['test_ood'] = self._load(dep_file=conf.test_ood, mode='test')
        return datasets

    def _load_dataset(self, dep_file, mode):
        log.info(f"Loading:{dep_file}")
        loader = ConllLoader(["raw_word", "lemma", "pos", "rel"], indexes=[1, 2, 4, 8])
        dataset = loader._load(dep_file)
        return dataset

    # deal with multihead
    def process_dataset(self, data):
        final_gold = []
        for sentence in data:
            gold_to_append = []
            for index, label in enumerate(sentence):
                if label == '_':
                    continue
                labels =  label.split("|")
                for label in labels:
                    parent, l = label.split(":")
                    gold_to_append.append([int(parent), index+1, l])
            final_gold.append(gold_to_append)
        return final_gold


    def _load(self, dep_file, mode):
        dataset = self._load_dataset(dep_file, mode)
        gold = self.process_dataset(dataset['rel'].content)
        dataset.add_field('raw_word', dataset['raw_word'])
        dataset.add_field('rel', gold)
        dataset.add_field('raw_raw_word', dataset['raw_word'])
        dataset.add_field("char", dataset['raw_word'])
        dataset.add_field("word", dataset['raw_word'])
        dataset.add_field("lemma", dataset['lemma'])
        dataset.add_field('id', [i for i in range(len(dataset))])
        dataset.add_seq_len("raw_word", 'seq_len')
        return dataset


    def _set_padder(self, datasets):
        set_padder(datasets, "rel", SpanLabelPadder())

    def build_fields(self, train_data):
        fields = {}
        fields['word'] = Field('word', pad=PAD, unk=UNK, bos=BOS, eos=EOS, lower=True, min_freq=self.conf.min_freq)
        fields['lemma'] = Field('lemma', pad=PAD, unk=UNK, bos=BOS, eos=EOS, lower=False, min_freq=7)
        fields['pos'] = Field('pos', pad=PAD, unk=UNK, bos=BOS, eos=EOS)
        fields['rel'] = SpanField('rel', unk=UNK)
        fields['char'] = SubwordField('char', pad=PAD, unk=UNK, bos=BOS, eos=EOS, fix_len=self.conf.fix_len)
        for name, field in fields.items():
            field.build(train_data[name])
        return fields




class SDPData_multiple(DataModuleBase):
    def __init__(self, conf):
        super(SDPData_multiple, self).__init__(conf)

    def get_inputs(self):
        inputs = ['seq_len', 'raw_word', 'id']
        if self.conf.use_pos:
            inputs.append('pos')
        if self.conf.use_lemma:
            inputs.append('lemma')
        if self.conf.use_char:
            inputs.append('char')
        return inputs

    def get_targets(self):
        return ['rel']

    def build_datasets(self):
        datasets = {}
        conf = self.conf
        datasets['train'] = self._load(dep_file=conf.train, mode='train')
        datasets['dev'] = self._load(dep_file=conf.dev, mode='dev')
        datasets['test'] = self._load(dep_file=conf.test_id, mode='test')
        datasets['test_ood'] = self._load(dep_file=conf.test_ood, mode='test')
        return datasets

    def _load_dataset(self, dep_file, mode):
        log.info(f"Loading:{dep_file}")
        loader = ConllLoader(["raw_word", "lemma", "pos", "rel"], indexes=[1, 2, 4, 8])
        dataset = loader._load(dep_file)
        return dataset

    # deal with multihead
    def process_dataset(self, data):
        final_gold = []
        for sentence in data:
            gold_to_append = []
            for index, label in enumerate(sentence):
                if label == '_':
                    continue
                labels =  label.split("|")
                for label in labels:
                    parent, l = label.split(":")
                    gold_to_append.append([int(parent), index+1, l])
            final_gold.append(gold_to_append)
        return final_gold


    def _load(self, dep_file, mode):
        dataset = self._load_dataset(dep_file, mode)
        gold = self.process_dataset(dataset['rel'].content)
        dataset.add_field('raw_word', dataset['raw_word'])
        dataset.add_field('rel', gold)
        dataset.add_field('raw_raw_word', dataset['raw_word'])
        dataset.add_field("char", dataset['raw_word'])
        dataset.add_field("word", dataset['raw_word'])
        dataset.add_field("lemma", dataset['lemma'])
        dataset.add_field('id', [i for i in range(len(dataset))])
        dataset.add_seq_len("raw_word", 'seq_len')
        return dataset


    def _set_padder(self, datasets):
        set_padder(datasets, "rel", SpanLabelPadder())

    def build_fields(self, train_data):
        fields = {}
        fields['word'] = Field('word', pad=PAD, unk=UNK, bos=BOS, eos=EOS, lower=True, min_freq=self.conf.min_freq)
        fields['lemma'] = Field('lemma', pad=PAD, unk=UNK, bos=BOS, eos=EOS, lower=False, min_freq=1)
        fields['pos'] = Field('pos', pad=PAD, unk=UNK, bos=BOS, eos=EOS)
        fields['rel'] = SpanField('rel', unk=UNK)
        fields['char'] = SubwordField('char', pad=PAD, unk=UNK, bos=BOS, eos=EOS, fix_len=self.conf.fix_len)
        for name, field in fields.items():
            field.build(train_data[name])
        return fields



    def test_dataloader(self):
        length = self.datasets['test'].get_field('seq_len').content
        length2 = self.datasets['test_ood'].get_field('seq_len').content
        sampler = get_sampler(lengths=length, max_tokens=self.conf.max_tokens_test,
                                n_buckets=self.conf.bucket_test, distributed=self.conf.distributed, evaluate=True)

        sampler2 = get_sampler(lengths=length2, max_tokens=self.conf.max_tokens_test,
                                n_buckets=self.conf.bucket_test, distributed=self.conf.distributed, evaluate=True)

        return [DataSetIter(self.datasets['test'], batch_size=1, sampler=None, as_numpy=False, num_workers=4,
                               pin_memory=False,drop_last=False, timeout=0, worker_init_fn=None,
                               batch_sampler=sampler),
                DataSetIter(self.datasets['test_ood'], batch_size=1, sampler=None, as_numpy=False, num_workers=4,
                               pin_memory=False,drop_last=False, timeout=0, worker_init_fn=None,
                               batch_sampler=sampler2)
                ]