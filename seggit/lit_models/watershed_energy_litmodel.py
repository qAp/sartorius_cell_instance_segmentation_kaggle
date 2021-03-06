
import importlib
import torch
import albumentations as albu
import pytorch_lightning as pl

from seggit.lit_models.losses import (WatershedEnergyLoss, 
                                      WatershedEnergyLoss1)



LR = 5e-4
WEIGHT_DECAY = 1e-6
OPTIMIZER = 'Adam'
ONE_CYCLE_TOTAL_STEPS = 25
LOSS = 'WatershedEnergyLoss1'



def _import_class(module_class_name):
    module_name, class_name = module_class_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class WatershedEnergyLitModel(pl.LightningModule):
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

    def forward(self, x):
        return self.model(x)

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
                                   weight_decay=WEIGHT_DECAY)
        if self.one_cycle_max_lr is None:
            return optimizer
        else:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.one_cycle_max_lr,
                total_steps=self.one_cycle_total_steps)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def training_step(self, batch, batch_idx):
        uvec, wngy, semg, area  = batch

        uvec = uvec.permute(0, 3, 1, 2)
        wngy = wngy.permute(0, 3, 1, 2)
        semg = semg.permute(0, 3, 1, 2)
        area = area.permute(0, 3, 1, 2)

        logits = self(uvec)

        loss = self.train_loss(logits, wngy, semg, area)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        uvec, wngy, semg, area = batch

        uvec = uvec.permute(0, 3, 1, 2)
        wngy = wngy.permute(0, 3, 1, 2)
        semg = semg.permute(0, 3, 1, 2)
        area = area.permute(0, 3, 1, 2)

        logits = self(uvec)

        loss = self.val_loss(logits, wngy, semg, area)
        self.log('val_loss', loss, prog_bar=True)

