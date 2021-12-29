
from os import get_terminal_size
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from seggit.data.config import WATERSHED_ENERGY_BINS


class SemSegLoss(nn.Module):
    def __init__(self, dice=0.5, bce=0.5):
        super().__init__()
        self.dice = dice
        self.bce = bce

        self.dice_loss = smp.utils.losses.DiceLoss()
        self.bce_loss = torch.nn.BCELoss()

    def forward(self, pr, gt):
        return (
            self.dice * self.dice_loss(pr, gt) +
            self.bce * self.bce_loss(pr, gt)
        )


class DirectionLoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, targ, mask, instance_area):
        '''
        pred [N, 2, H, W]
        targ [N, 2, H, W]
        mask [N, 1, H, W]
        instance_area [N, 1, H, W]
        '''
        batch_size = pred.shape[0]

        pred = pred.permute(1, 0, 2, 3).reshape(2, -1)
        targ = targ.permute(1, 0, 2, 3).reshape(2, -1)
        mask = mask.permute(1, 0, 2, 3).reshape(1, -1).type(torch.bool)
        instance_area = instance_area.permute(1, 0, 2, 3).reshape(1, -1)

        dotprod = (pred * targ).sum(dim=0, keepdims=True)

        angle_squared = (dotprod
                         .clamp(min=-1 + self.epsilon, max=1 - self.epsilon)
                         .acos()
                         .pow(2)
                         )

        instance_area[instance_area == 0] = instance_area.max()
        weights = 1 / instance_area.sqrt()

        weighted_sum = (weights * angle_squared).sum()

        # return weighted_sum / (len(mask.ravel()) + 1)        
        return weighted_sum / (mask.sum() + 1)

 
class WatershedEnergyLoss(nn.Module):
    '''
    Issues:
    1. Not sure what the weights for different energy levels should be.
    '''
    def __init__(self):
        super().__init__()
        self.n_energy = len(WATERSHED_ENERGY_BINS) + 1
        self.logsoftmax = nn.LogSoftmax(dim=1)
        weight_energy = torch.arange(self.n_energy, 0, -1).type(torch.float32)        
        self.nlloss = nn.NLLLoss(weight=weight_energy)

    def forward(self, logits, energy, semseg, area):
        logits = logits.permute(0, 2, 3, 1).reshape(-1, self.n_energy)
        semseg = semseg.view(-1)
        area = area.view(-1)
        energy = energy.view(-1)

        mask = semseg.type(torch.bool)
        logits = logits[mask]
        area = area[mask]
        energy = energy[mask].type(torch.long)

        logp = self.logsoftmax(logits)

        weight_pixel = 1 / area.sqrt()
        loss = self.nlloss(weight_pixel[..., None] * logp, energy)

        return loss


class WatershedEnergyLoss1(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-25
        self.num_classes = len(WATERSHED_ENERGY_BINS) + 1
        self.register_buffer(
            'cs_scaling',
            torch.tensor([3.0, 3.0, 3.0, 2.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                          1.0, 1.0, 1.0, 1.0, 1.0])
        )

    def forward(self, logits, wngy, semg, area):
        '''
        Args:
            logits (N, 17, H, W)
            wngy (N, 1, H, W)
            semg (N, 1, H, W)
            area (N, 1, H, W)
        '''
        logits = logits.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        wngy = wngy.permute(0, 2, 3, 1).reshape(-1, 1)
        semg = semg.permute(0, 2, 3, 1).reshape(-1).type(torch.bool)
        area = area.permute(0, 2, 3, 1).reshape(-1, 1)

        logits = logits[semg, :]
        wngy = wngy[semg]
        area = area[semg]

        weight = 1 / area.sqrt()

        predSoftmax = F.softmax(logits, dim=1)
        gt = F.one_hot(wngy.squeeze(), num_classes=self.num_classes)

        ll = (
            gt * predSoftmax.clamp(min=self.epsilon).log() +
            (1 - gt) * (1 - predSoftmax).clamp(min=self.epsilon).log()
        )

        cs = - (ll * weight * self.cs_scaling).sum()

        return cs / (weight.sum() + 1)


def dice_coef(y_pred, y_true, smooth=1e-3):
    '''
    Args:
        y_pred (N, H, W)    
        y_true (N, H, W)
    '''
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    intersection = (y_true * y_pred).sum()
    coef = (2*intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)
    
    return coef


def dice_coef_loss(y_pred, y_true):
    '''
    Args:
        y_pred (N, H, W)
        y_true (N, H, W)        
    '''    
    return 1 - dice_coef(y_pred, y_true)


class SoftmaxDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, logits, semseg):
        '''
        Args:
            logits (N, C, H, W): Softmax probability
            semseg (N, C, H, W): Semantic segmentation (one-hot).
            
        Notes:
            C=0 Cell
            C=1 Overlapping cell wall
            C=2 Background
        '''
        loss = (
            .6 * self.cross_entropy(logits, semseg.argmax(dim=1)) + 
            .2 * dice_coef_loss(logits[:, 0, ...], semseg[:, 0, ...]) +
            .2 * dice_coef_loss(logits[:, 1, ...], semseg[:, 1, ...])
        )
        return loss