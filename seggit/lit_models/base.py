
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp


LR = 1e-4
OPTIMIZER = 'Adam'
ONE_CYCLE_TOTAL_STEPS = 100


class BaseLitModel(pl.LightningModule):
    def __init__(self, model, args=None):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.model = model

        self.lr = self.args.get('lr', LR)
        optimizer = self.args.get('optimizer', OPTIMIZER)
        self.optimizer = getattr(torch.optim, optimizer)

        self.one_cycle_max_lr = self.args.get('one_cycle_max_lr', None)
        self.one_cycle_total_steps = self.args.get('one_cycle_total_steps', 
                                                   ONE_CYCLE_TOTAL_STEPS)

        self.loss_fn = smp.losses.FocalLoss(mode='binary')

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--lr', type=float, default=LR)
        parser.add_argument('--optimizer', type=str, default=OPTIMIZER)
        parser.add_argument('--one_cycle_total_steps', type=int, 
                            default=ONE_CYCLE_TOTAL_STEPS)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)

        if self.one_cycle_max_lr is None:
            return optimizer

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.one_cycle_max_lr, 
            total_steps=self.one_cycle_total_steps)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)

        self.log('train_loss', loss, 
                 on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)

        self.log('val_loss', loss, 
                 on_step=False, on_epoch=True, prog_bar=True)