
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

        self.loss_fn = smp.utils.losses.DiceLoss()
        self.train_iou5 = smp.utils.metrics.IoU(threshold=0.5)
        self.val_iou3 = smp.utils.metrics.IoU(threshold=0.3)
        self.val_iou5 = smp.utils.metrics.IoU(threshold=0.5)
        self.val_iou7 = smp.utils.metrics.IoU(threshold=0.7)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--lr', type=float, default=LR)
        parser.add_argument('--optimizer', type=str, default=OPTIMIZER)
        parser.add_argument('--one_cycle_max_lr', type=float, default=None)
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
        loss = self.loss_fn(y_pred.squeeze(1), y)
        train_iou5 = self.train_iou5(y_pred.squeeze(1), y)

        self.log('train_loss', loss)
        self.log('train_iou5', train_iou5, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred.squeeze(1), y)

        iou3 = self.val_iou3(y_pred.squeeze(1), y)
        iou5 = self.val_iou5(y_pred.squeeze(1), y)
        iou7 = self.val_iou7(y_pred.squeeze(1), y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_iou3', iou3, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou5', iou5, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou7', iou7, on_step=False, on_epoch=True, prog_bar=True)