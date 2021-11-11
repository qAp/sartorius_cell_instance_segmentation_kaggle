import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp



class BaseLitModel(pl.LightningModule):
    def __init__(self, model, args=None):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.model = model

        self.lr = 1e-3
        self.optimizer = torch.optim.Adam

        self.loss_fn = smp.losses.FocalLoss(mode='binary')


    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.OneCyleLR(optimizer, max_lr=1e-2)
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