
import torch.nn as nn

import torch
import torch.nn as nn
from supar.modules import LSTM, MLP, BertEmbedding, Biaffine, CharLSTM, Triaffine
from supar.modules.dropout import IndependentDropout, SharedDropout
from supar.modules.treecrf import CRFConstituency
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pytorch_lightning as pl
import hydra
# from .model_utils import *
from supar.utils.transform import Tree
import nltk
import unicodedata
import hydra
from pytorch_lightning.core.decorators import auto_move_data
import logging
import copy
log = logging.getLogger(__name__)
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from omegaconf import OmegaConf, open_dict
from ..model.metric import *
import copy

## To handle multiple test set.  SDP (id and ood).

class Runner(pl.LightningModule):
    ## conf: model's config
    ## config: the whole config.... use for initialize the optimizer and scheduler;;;

    def __init__(self, cfg, fields, **kwargs):
        super(Runner, self).__init__()
        # self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.fields = fields

        model_cfg = cfg.model
        optim_cfg = cfg.optim

        self.cfg = cfg
        self.model = hydra.utils.instantiate(model_cfg.target, conf=model_cfg, fields=fields,  _recursive_=False)
        self.optimizer_cfg = optim_cfg
        self.model_cfg = model_cfg
        self.metric_val = self.model.metric
        self.metric_test_id = copy.deepcopy(self.model.metric)
        self.metric_test_ood = copy.deepcopy(self.model.metric)

        self.metric_dev_hist = []
        self.metric_test_hist =  []
        self.metric_test_ood_hist =  []
        self.best_dev_metric_epoch = -1
        # self.save_hyperparameters(OmegaConf.to_container(self.cfg, resolve=True))
        self.test = False

    def forward(self, x):
        return self.model(x)

    @property
    def result(self):
        return self.metric_test_hist[self.best_dev_metric_epoch]['score']

    @auto_move_data
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.model.get_loss(x, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        info = self.model.decode(x, y)
        self.metric_val.update(info)

    def on_fit_start(self) -> None:
        self.metric_val.to(self.device)
        self.metric_test_id.to(self.device)
        self.metric_test_ood.to(self.device)

    def on_train_epoch_end(self, outputs) -> None:
        self.trainer.test(ckpt_path=None)

    def on_validation_epoch_start(self) -> None:
        self.metric_val.reset()

    def validation_epoch_end(self, outputs) -> None:
        mode = 'val'
        result = self.metric_val.compute(mode='val')
        self.log_dict({f'{mode}/' + k: v for k, v in result.items()})
        self.print('\n')
        self.print(f'[Epoch {self.current_epoch}]\t{mode} \t' + '\t'.join(f'{k}={v:.4f}' for k, v in result.items()))
        self.update_metric(result=result, mode='val')

    def print(self, msg):
        if self.trainer.is_global_zero:
            log.info(msg)

    def update_metric(self, result, mode='val'):

        if mode == 'val':
            self.metric_dev_hist.append(result)
            if result['score'] >= self.metric_dev_hist[self.best_dev_metric_epoch]['score']:
                self.best_dev_metric_epoch = self.current_epoch
                # upgrade the best result file. Notice that 我们先测试再验证，所以根据验证集选择的时候，测试的output file已经出来了。
                if self.metric_val.cfg.write_result_to_file:
                    os.system(f"cp {self.metric_val.prefix}_output_valid.txt  {self.metric_val.prefix}_output_best_valid.txt")
                    os.system(f"cp {self.metric_val.prefix}_output_test.txt  {self.metric_val.prefix}_output_best_test.txt")
                    os.system(f"cp {self.metric_val.prefix}_output_test_ood.txt  {self.metric_val.prefix}_output_best_test_ood.txt")

            self.print(
            f'[best dev: {self.best_dev_metric_epoch}]\t DEV \t' + '\t'.join(f'{k}={v:.4f}'
                                            for k, v in self.metric_dev_hist[self.best_dev_metric_epoch].items())
            )

            best_result = self.metric_test_hist[self.best_dev_metric_epoch]
            self.print(
            f'[best test: {self.best_dev_metric_epoch}]\t TEST \t' + '\t'.join(f'{k}={v:.4f}'
                                                                              for k, v in best_result.items())
            )
            best_result = self.metric_test_ood_hist[self.best_dev_metric_epoch]
            self.print(
                f'[best test_ood: {self.best_dev_metric_epoch}]\t TEST \t' + '\t'.join(f'{k}={v:.4f}'
                                                                              for k, v in best_result.items())
            )
            self.log_dict({'best_epoch': self.best_dev_metric_epoch})
            self.log_dict({f'final/' + k: v for k, v in best_result.items()})

        elif mode == 'test':
            self.metric_test_hist.append(result)

        elif mode == 'test_ood':
            self.metric_test_ood_hist.append(result)

        else:
            raise NameError

    def configure_optimizers(self):
        hparams = self.optimizer_cfg
        # lr_rate: 用来放大encoder的learning rate， 如果存在，那么用的就是finetuning的模式。
        if hparams.get("lr_rate") is not None:
            if hparams.only_embeder:
                optimizer = AdamW(
                    [{'params': c.parameters(), 'lr': hparams.lr * (1 if n == 'embeder' else hparams.lr_rate)}
                     for n, c in self.model.named_children()], hparams.lr
                )
                log.info(f"Embeder has learning rate:{  hparams.lr},Encoder has learning rate:{hparams.lr * hparams.lr_rate}, Scorer has lr:{hparams.lr * hparams.lr_rate}"
                         )

            else:
                optimizer = torch.optim.Adam(
                    [{'params': c.parameters(), 'lr': hparams.lr * (1 if n == 'embeder' or n == 'encoder' else hparams.lr_rate)}
                     for n, c in self.model.named_children()], hparams.lr, betas=(0.9, 0.9)
                )
                log.info(f"Embeder has learning rate:{hparams.lr},Encoder has learning rate:{hparams.lr}, Scorer has lr:{hparams.lr * hparams.lr_rate}"
                         )

            if hparams.scheduler_type == 'linear_warmup':
                scheduler = get_linear_schedule_with_warmup(optimizer, 211231123, 31212312312)
                scheduler = {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
                log.info("Using huggingface transformer linear-warmup scheduler.")

            elif hparams.scheduler_type == 'constant_warmup':
                scheduler = get_constant_schedule_with_warmup(optimizer, 211231123)
                scheduler = {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
                log.info("Using huggingface transformer constant_warmup scheduler.")

            return [optimizer], [scheduler]

        else:
            opt = hydra.utils.instantiate(
                hparams.optimizer, params=self.parameters(), _convert_='all'
            )

            if hparams.use_lr_scheduler:
                if hparams.lr_scheduler._target_ == 'torch.optim.lr_scheduler.ExponentialLR':
                    scheduler =  torch.optim.lr_scheduler.ExponentialLR(opt, gamma=.75 ** (1 / 5000))
                    scheduler = {
                        'scheduler': scheduler,
                        'interval': 'step',  # or 'epoch'
                        'frequency': 1
                    }
                    log.info("Using ExponentialLR")

                else:
                    raise NotImplementedError

                return [opt], [scheduler]
            return opt


    def test_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        info = self.model.decode(x, y)
        # test_id dataloader.
        if dataloader_idx == 0:
            self.metric_test_id.update(info)
        # test_ood dataloader
        elif dataloader_idx == 1:
            self.metric_test_ood.update(info)
        else:
            raise ValueError

    def on_test_epoch_start(self):
        self.metric_test_id.reset()
        self.metric_test_ood.reset()

    def test_epoch_end(self, outputs):
        result_id = self.metric_test_id.compute(mode='test')
        result_ood = self.metric_test_ood.compute(mode='test_ood')
        self.log_dict({f'test_id/' + k: v for k, v in result_id.items()})
        self.log_dict({f'test_ood/' + k: v for k, v in result_id.items()})
        self.print('\n')
        self.print(f'[Epoch {self.current_epoch}]\t test_id  \t' + '\t'.join(f'{k}={v:.4f}' for k, v in result_id.items()))
        self.print(f'[Epoch {self.current_epoch}]\t test_ood \t' + '\t'.join(f'{k}={v:.4f}' for k, v in result_ood.items()))
        self.update_metric(result=result_id, mode='test')
        self.update_metric(result=result_ood, mode='test_ood')
