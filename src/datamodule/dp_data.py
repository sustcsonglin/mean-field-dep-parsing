import pytorch_lightning as pl
import os
from supar.utils.common import *
import pickle
from .dm_util.fields import SubwordField, Field
from .dm_util.padder import *
import logging
log = logging.getLogger(__name__)
from .base import DataModuleBase
from fastNLP.io.loader.conll import ConllLoader
from .dm_util.util import *
import tempfile
from supar.utils.transform import CoNLL
from pathlib import Path


class DepDataBase(DataModuleBase):
    def __init__(self, conf):
        super(DepDataBase, self).__init__(conf)

    def get_inputs(self):

        inputs = ['seq_len', 'raw_word', 'id']

        if self.conf.ignore_punct:
            inputs.append('is_punct')

        return inputs

    def get_targets(self):
        return ['head', 'rel']

    def build_datasets(self):
        datasets = {}
        conf = self.conf
        datasets['train'] = self._load(dep_file=conf.train_dep, mode='train')
        datasets['dev'] = self._load(dep_file=conf.dev_dep, mode='dev')
        datasets['test'] = self._load(dep_file=conf.test_dep, mode='test')
        return datasets

    def _load_dataset(self, dep_file, mode):
        raise NotImplementedError

    def _load(self, dep_file, mode):
        #build dependency, make sure your file is of conll format.
        dataset = self._load_dataset(dep_file, mode)
        dataset['head'].int()

        #Identify the projectiveness of the tree and identify all punctuations in the sentence: sometimes we need to omit the punc. during evaluations
        if self.conf.force_projectivity:
            valid = [isProjective(head) for head in dataset['head'].content]
            dataset.add_field("valid", valid)

        punct = [[is_punctuation(word, pos) for word, pos in zip(words, poses)] for words, poses in zip(dataset['raw_word'].content, dataset['pos'].content)]
        dataset.add_field("is_punct", punct)

        # clean word: remove all (-RHS) and numbers.
        if self.conf.clean_word:
            dataset.apply_field(clean_word, 'raw_word', 'raw_word')

        dataset.add_field('raw_word', dataset['raw_word'])
        dataset.add_field('raw_raw_word', dataset['raw_word'])
        dataset.add_field("char", dataset['raw_word'])
        dataset.add_field("word", dataset['raw_word'])
        dataset.add_field('id', [i for i in range(len(dataset))])
        dataset.add_seq_len("raw_word", 'seq_len')
        return dataset

    def _set_padder(self, datasets):
        set_padder(datasets, "head", DepPadder())
        set_padder(datasets, "rel", DepPadder())

    def build_fields(self, train_data):
        fields = {}
        fields['word'] = Field('word', pad=PAD, unk=UNK, bos=BOS, eos=EOS, lower=True, min_freq=self.conf.min_freq)
        fields['pos'] = Field('pos', pad=PAD, unk=UNK, bos=BOS, eos=EOS)
        fields['rel'] = Field('rel', pad=PAD, unk=UNK)
        fields['char'] = SubwordField('char', pad=PAD, unk=UNK, bos=BOS, eos=EOS, fix_len=self.conf.fix_len)
        for name, field in fields.items():
            field.build(train_data[name])
        return fields


# PTB, CTB
class DepData(DepDataBase):
    def __init__(self, conf):
        super(DepData, self).__init__(conf)

    def _load_dataset(self, dep_file, mode):
        log.info(f"Loading:{dep_file}")
        loader = ConllLoader(["raw_word", "pos", "head", "rel"], indexes=[1, 3, 6, 7])
        dataset = loader._load(dep_file)
        return dataset

# German
class Conll09(DepDataBase):
    def __init__(self, conf):
        super(Conll09, self).__init__(conf)

    def _load_dataset(self, dep_file, mode):
        log.info(f"Loading:{dep_file}")
        # 4 is gold pos, 5 is predicted pos. We follow RNGTr to use predicted pos here.
        loader = ConllLoader(["raw_word", "pos", "head", "rel"], indexes=[1, 5, 8, 10])
        dataset = loader._load(dep_file)
        return dataset

# UD
class DepUD(DepDataBase):
    def __init__(self, conf):
        super(DepUD, self).__init__(conf)

    def _load_dataset(self, path, mode):
        tmpdir = '/dev/shm' if os.path.exists('/dev/shm') else None
        # remove comments, multiword and empty nodes
        # https://universaldependencies.org/format.html#words-tokens-and-empty-nodes
        input_filename = tempfile.mktemp(dir=tmpdir)
        command = r"awk -F '\t' '$1~/^[0-9]+$/ {print $0} $0~/^$/ {print}' " + path + f" > {input_filename}"
        log.debug(f'Intermediate file is {input_filename}')
        os.system(command)
        log.info(f'Converting {path}')
        path = input_filename
        loader = ConllLoader(['word_id', "raw_word", "pos", "head", "rel"], indexes=[0, 1, 3, 6, 7], dropna=False, sep='\t')
        dataset = loader._load(path)
        return dataset
