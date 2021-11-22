
import torch
import pytorch_lightning as pl
from seggit.lit_models.losses import DirectionLoss


LR = 1e-5
OPTIMIZER = 'Adam'
ONE_CYCLE_TOTAL_STEPS = 100


class InstanceDirectionLitModel(pl.LightningModule):
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

        self.train_loss = DirectionLoss()
        self.val_loss = DirectionLoss()

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--lr', type=float, default=LR)
        parser.add_argmuent('--optimizer', type=str, default=OPTIMIZER)
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
                max_lr=self.one_cycle_max_lr,
                total_steps=self.one_cycle_total_steps)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def training_step(self, batch, batch_idx):
        img, uvec, semseg, area = batch

        img = img.permute(0, 3, 1, 2)
        uvec = uvec.permute(0, 3, 1, 2)
        semseg = semseg.permute(0, 3, 1, 2)
        area = area.permute(0, 3, 1, 2)

        img_in = semseg * img  

        logits = self(img_in)

        loss = self.train_loss(logits, uvec, semseg, area)

        self.log('train_loss', loss)

    def validation_step(self, batch, batch_idx):
        img, uvec, semseg, area = batch 

        img = img.permute(0, 3, 1, 2)
        uvec = uvec.permute(0, 3, 1, 2)
        semseg = semseg.permute(0, 3, 1, 2)
        area = area.permute(0, 3, 1, 2)

        img_in = semseg * img

        logits = self(img_in)

        loss = self.val_loss(logits, uvec, semseg, area)

        self.log('val_loss', loss)
