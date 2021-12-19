import importlib
import torch
import albumentations as albu
import pytorch_lightning as pl

from seggit.lit_models.losses import (WatershedEnergyLoss, 
                                      WatershedEnergyLoss1)
from seggit.lit_models.metrics import NCorrectPredictions



LR = 5e-4
OPTIMIZER = 'Adam'
ONE_CYCLE_TOTAL_STEPS = 100
LOSS = 'WatershedEnergyLoss1'



def _import_class(module_class_name):
    module_name, class_name = module_class_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class WatershedLitModel(pl.LightningModule):
    def __init__(self, model, args=None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        self.lr = self.args.get('lr', LR)
        self.one_cycle_max_lr = self.args.get(
            'one_cycle_max_lr', None
        )
        self.one_cycle_total_steps = self.args.get(
            'one_cycle_total_steps', ONE_CYCLE_TOTAL_STEPS
        )
        optimizer = self.args.get('optimizer', OPTIMIZER)
        self.optimizer = getattr(torch.optim, optimizer)

        loss = self.args.get('loss', LOSS)
        loss_class = _import_class(f'seggit.lit_models.losses.{loss}')
        self.train_loss = loss_class()
        self.val_loss = loss_class()
        self.metric_func = NCorrectPredictions()

    def forward(self, img, semg):
        return self.model(img, semg)

    @staticmethod
    def add_argparse_args(parser):
        add = parser.add_argument
        add('--lr', type=float, default=LR)
        add('--optimizer', type=str, default=OPTIMIZER)
        add('--one_cycle_max_lr', type=float, default=None)
        add('--one_cycle_total_steps', type=int, default=ONE_CYCLE_TOTAL_STEPS)
        add('--loss', type=str, default=LOSS)


    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), 
                                   lr=self.lr, 
                                   weight_decay=1e-5)
        if self.one_cycle_max_lr is None:
            return optimizer
        else:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.one_cycle_max_lr,
                total_steps=self.one_cycle_total_steps)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def training_step(self, batch, batch_idx):
        img, wngy, semg, area  = batch

        img = img.permute(0, 3, 1, 2)
        wngy = wngy.permute(0, 3, 1, 2)
        semg = semg.permute(0, 3, 1, 2)
        area = area.permute(0, 3, 1, 2)

        logits = self(img, semg)

        loss = self.train_loss(logits, wngy, semg, area)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, wngy, semg, area = batch

        img = img.permute(0, 3, 1, 2)
        wngy = wngy.permute(0, 3, 1, 2)
        semg = semg.permute(0, 3, 1, 2)
        area = area.permute(0, 3, 1, 2)

        logits = self(img, semg)

        loss = self.val_loss(logits, wngy, semg, area)
        self.log('val_loss', loss, prog_bar=True)

        metric = self.metric_func(logits, wngy, semg)
        self.log('val_ncorrect', metric, 
                 on_step=False, on_epoch=True, prog_bar=True)