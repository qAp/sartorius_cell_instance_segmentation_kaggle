

import torch
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import albumentations as albu
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

import seggit.lit_models.losses


ARCH = 'Unet'
ENCODER_NAME = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
LOSS = 'SemSegLoss'
LR = 1e-4
OPTIMIZER = 'Adam'
ONE_CYCLE_TOTAL_STEPS = 400
ONE_CYCLE_MAX_LR = None
STEP_LR_GAMMA = None


class SemSegLitModel(pl.LightningModule):
    def __init__(self, model, args=None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        loss = self.args.get('loss', LOSS)
        loss_class = getattr(seggit.lit_models.losses, loss)
        self.loss_func = loss_class()

        self.iou55 = smp.utils.metrics.IoU(threshold=0.55)
        self.iou65 = smp.utils.metrics.IoU(threshold=0.65)
        self.iou75 = smp.utils.metrics.IoU(threshold=0.75)
        self.iou85 = smp.utils.metrics.IoU(threshold=0.85)
        self.iou95 = smp.utils.metrics.IoU(threshold=0.95)

        optimizer = self.args.get('optimizer', OPTIMIZER)
        self.optimizer = getattr(torch.optim, optimizer)
        self.lr = self.args.get('lr', LR)
        self.one_cycle_total_steps = self.args.get('one_cycle_total_steps', 
                                                   ONE_CYCLE_TOTAL_STEPS)
        self.one_cycle_max_lr = self.args.get('one_cycle_max_lr', 
                                              ONE_CYCLE_MAX_LR)                                           
        self.step_lr_gamma = self.args.get('step_lr_gamma', STEP_LR_GAMMA)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def add_argparse_args(parser):
        add = parser.add_argument
        add('--arch', type=str, default=ARCH)
        add('--encoder_name', type=str, default=ENCODER_NAME)
        add('--encoder_weights', type=str, default=ENCODER_WEIGHTS)
        add('--loss', type=str, default=LOSS)
        add('--optimizer', type=str, default=OPTIMIZER)
        add('--lr', type=float, default=LR)
        add('--one_cycle_total_steps', type=int, default=ONE_CYCLE_TOTAL_STEPS)
        add('--one_cycle_max_lr', type=float, default=ONE_CYCLE_MAX_LR)
        add('--step_lr_gamma', type=float, default=STEP_LR_GAMMA)
        return parser

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)

        if self.one_cycle_max_lr is not None:
            lr_scheduler = OneCycleLR(optimizer, 
                                      total_steps=self.one_cycle_total_steps,
                                      max_lr=self.one_cycle_max_lr)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        elif self.step_lr_gamma:
            lr_scheduler = StepLR(optimizer, 
                                  step_size=30, 
                                  gamma=self.step_lr_gamma)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        img, semseg = batch

        img = img.permute(0, 3, 1, 2)
        semseg = semseg.permute(0, 3, 1, 2)

        logits = self(img)

        loss = self.loss_func(logits, semseg)
        self.log('train_loss', loss)

        self.log('train_iou95', self.iou95(logits, semseg), 
                 on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, semseg = batch

        img = img.permute(0, 3, 1, 2)
        semseg = semseg.permute(0, 3, 1, 2)

        logits = self(img)

        kwargs = dict(on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', self.loss_func(logits, semseg), **kwargs)
        self.log('val_iou55', self.iou55(logits, semseg), **kwargs)
        self.log('val_iou65', self.iou65(logits, semseg), **kwargs)
        self.log('val_iou75', self.iou75(logits, semseg), **kwargs)
        self.log('val_iou85', self.iou85(logits, semseg), **kwargs)
        self.log('val_iou95', self.iou95(logits, semseg), **kwargs)





    

