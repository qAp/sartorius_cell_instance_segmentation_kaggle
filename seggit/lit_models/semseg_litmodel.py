
import torch
import albumentations as albu
import pytorch_lightning as pl

from seggit.lit_models.losses import SemSegLoss


ARCH = 'Unet'
ENCODER_NAME = 'resnet34'
LR = 1e-4
OPTIMIZER = 'Adam'


class SemSegLitModel(pl.LightningModule):
    def __init__(self, model, args=None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else args

        self.train_loss = SemSegLoss()
        self.val_loss = SemSegLoss()

        self.lr = self.args.get('lr', LR)
        optimizer = self.args.get('optimizer', OPTIMIZER)
        self.optimizer = getattr(torch.optim, optimizer)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def add_argparse_args(parser):
        add = parser.add_argument
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

        loss = self.train_loss(logits, semseg)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        img, semseg = batch

        img = img.permute(0, 3, 1, 2)
        semseg = semseg.permute(0, 3, 1, 2)

        logits = self(img)

        loss = self.val_loss(logits, semseg)

        self.log('val_loss', loss)

    

