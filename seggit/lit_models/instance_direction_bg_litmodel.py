
import torch
import pytorch_lightning as pl
from seggit.lit_models.losses import DirectionLoss


LR = 1e-5
OPTIMIZER = 'Adam'
ONE_CYCLE_TOTAL_STEPS = 100


class InstanceDirectionMockLitModel(pl.LightningModule):
    def __init__(self, model=None, args=None):
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

        self.train_loss = DirectionLoss()
        self.val_loss = DirectionLoss()

    def forward(self, img, semg):
        return self.model(img, semg)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--lr', type=float, default=LR)
        parser.add_argument('--optimizer', type=str, default=OPTIMIZER)
        parser.add_argument('--one_cycle_max_lr', type=float, default=None)
        parser.add_argument('--one_cycle_total_steps', 
                            type=int, default=ONE_CYCLE_TOTAL_STEPS)

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
        img, uvec, semg, area = batch

        img = img.permute(0, 3, 1, 2)
        uvec = uvec.permute(0, 3, 1, 2)
        semg = semg.permute(0, 3, 1, 2)
        area = area.permute(0, 3, 1, 2)

        logits = self(img, semg)

        loss = self.train_loss(logits, uvec, semg, area)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, uvec, semg, area = batch 

        img = img.permute(0, 3, 1, 2)
        uvec = uvec.permute(0, 3, 1, 2)
        semg = semg.permute(0, 3, 1, 2)
        area = area.permute(0, 3, 1, 2)

        logits = self(img, semg)

        loss = self.val_loss(logits, uvec, semg, area)

        self.log('val_loss', loss, prog_bar=True)
