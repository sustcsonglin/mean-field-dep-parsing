import torch.nn as nn
import torch
from supar.modules.char_lstm import CharLSTM
from supar.modules import TransformerEmbedding
from src.model.module.ember.ext_embedding import ExternalEmbeddingSupar
import copy

class Embeder(nn.Module):
    def __init__(self, conf, fields):
        super(Embeder, self).__init__()
        self.conf = conf

        if 'pos' in fields.inputs:
            self.pos_emb = nn.Embedding(fields.get_vocab_size("pos"), conf.n_pos_embed)
        else:
            self.pos_emb = None

        if 'lemma' in fields.inputs:
            self.lemma_emb = nn.Embedding(fields.get_vocab_size("lemma"), conf.n_pos_embed)
        else:
            self.lemma_emb = None

        if 'char' in fields.inputs:
            self.char_emb = CharLSTM(n_chars=fields.get_vocab_size('char'),
                                       n_embed=conf.n_char_embed,
                                       n_out=conf.n_char_out,
                                      pad_index=fields.get_pad_index('char'),
                                        input_dropout=conf.char_input_dropout)
        else:
            self.char_emb = None

        if 'bert' in fields.inputs:
            self.bert_emb =  TransformerEmbedding(model=fields.get_bert_name(),
                                            n_layers=conf.n_bert_layers,
                                            n_out=conf.n_bert_out,
                                            pad_index=fields.get_pad_index("bert"),
                                            dropout=conf.mix_dropout,
                                            requires_grad=conf.finetune,
                                            use_projection=conf.use_projection,
                                            use_scalarmix=conf.use_scalarmix)

        else:
            self.bert_emb = None


        if 'word' in fields.inputs:
            ext_emb = fields.get_ext_emb()
            if ext_emb:
                self.word_emb =  copy.deepcopy(ext_emb)
            else:
                self.word_emb = nn.Embedding(num_embeddings=fields.get_vocab_size('word'),
                                             embedding_dim=conf.n_embed)
        else:
            self.word_emb = None


    def forward(self, ctx):
        emb = {}

        if self.pos_emb:
            emb['pos'] = self.pos_emb(ctx['pos'])

        if self.word_emb:
            emb['word'] = self.word_emb(ctx['word'])

        if self.char_emb:
            emb['char'] = self.char_emb(ctx['char'])

        if self.lemma_emb:
            emb['lemma'] = self.lemma_emb(ctx['lemma'])

        if self.bert_emb:
            emb['bert'] = self.bert_emb(ctx['bert'])

        ctx['embed'] = emb


    def get_output_dim(self):

        size = 0

        if self.pos_emb:
            size += self.conf.n_pos_embed

        if self.lemma_emb:
            size += self.conf.n_pos_embed

        if self.word_emb:
            if isinstance(self.word_emb, nn.Embedding):
                size += self.conf.n_embed
            else:
                size += self.word_emb.get_dim()

        if self.char_emb:
            size += self.conf.n_char_out

        if self.bert_emb:
            size += self.bert_emb.n_out

        return size



