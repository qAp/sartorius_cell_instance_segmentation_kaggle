
import torch
import albumentations as albu
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from seggit.lit_models.losses import SemSegLoss


ARCH = 'Unet'
ENCODER_NAME = 'resnet34'
LR = 1e-4
OPTIMIZER = 'Adam'


class SemSegLitModel(pl.LightningModule):
    def __init__(self, model, args=None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        self.loss_func = SemSegLoss() 
        self.val_iou55 = smp.utils.metrics.IoU(threshold=0.55)
        self.val_iou65 = smp.utils.metrics.IoU(threshold=0.65)
        self.val_iou75 = smp.utils.metrics.IoU(threshold=0.75)
        self.val_iou85 = smp.utils.metrics.IoU(threshold=0.85)
        self.val_iou95 = smp.utils.metrics.IoU(threshold=0.95)

        self.lr = self.args.get('lr', LR)
        optimizer = self.args.get('optimizer', OPTIMIZER)
        self.optimizer = getattr(torch.optim, optimizer)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def add_argparse_args(parser):
        add = parser.add_argument
        add('--lr', type=float, default=LR)
        add('--optimizer', type=str, default=OPTIMIZER)
        add('--arch', type=str, default=ARCH)
        add('--encoder_name', type=str, default=ENCODER_NAME)
        return parser

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        img, semseg = batch

        img = img.permute(0, 3, 1, 2)
        semseg = semseg.permute(0, 3, 1, 2)

        logits = self(img)

        loss = self.loss_func(logits, semseg)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        img, semseg = batch

        img = img.permute(0, 3, 1, 2)
        semseg = semseg.permute(0, 3, 1, 2)

        logits = self(img)

        kwargs = dict(on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', self.loss_func(logits, semseg), **kwargs)
        self.log('val_iou55', self.val_iou55(logits, semseg), **kwargs)
        self.log('val_iou65', self.val_iou65(logits, semseg), **kwargs)
        self.log('val_iou75', self.val_iou75(logits, semseg), **kwargs)
        self.log('val_iou85', self.val_iou85(logits, semseg), **kwargs)
        self.log('val_iou95', self.val_iou95(logits, semseg), **kwargs)





    

