
import torch
import albumentations as albu
import pytorch_lightning as pl

from seggit.lit_models.losses import WatershedEnergyLoss



LR = 5e-4
OPTIMIZER = 'Adam'


class WatershedEnergyLitModel(pl.LightningModule):
    def __init__(self, model, args=None):
        self.model = model

        self.args = vars(args) if args is not None else {}

        self.lr = self.args.get('lr', LR)
        optimizer = self.args.get('optimizer', OPTIMIZER)
        self.optimizer = getattr(torch.optim, optimizer)

        self.train_loss = WatershedEnergyLoss()
        self.val_loss = WatershedEnergyLoss()

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--lr', type=float, default=LR)
        parser.add_argument('--optimizer', type=str, default=OPTIMIZER)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters(), 
                                   lr=self.lr,
                                   weight_decay=1e-6)
        return optimizer

    def training_step(self, batch, batch_idx):
        semseg, area, uvec, energy = batch

        semseg = semseg.permute(0, 3, 1, 2)
        area = area.permute(0, 3, 1, 2)
        uvec = uvec.permute(0, 3, 1, 2)
        energy = energy.permute(0, 3, 1, 2)

        logits = self(uvec)

        loss = self.train_loss(logits, energy, semseg, area)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        semseg, area, uvec, energy = batch

        semseg = semseg.permute(0, 3, 1, 2)
        area = area.permute(0, 3, 1, 2)
        uvec = uvec.permute(0, 3, 1, 2)
        energy = energy.permute(0, 3, 1, 2)

        logits = self(uvec)

        loss = self.val_loss(logits, energy, semseg, area)
        self.log('val_loss', loss)

