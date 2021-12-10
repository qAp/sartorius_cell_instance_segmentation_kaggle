
import torch
import torch.nn as nn
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

        pred = pred.permute(1, 0, 2, 3).view(2, -1)
        targ = targ.permute(1, 0, 2, 3).view(2, -1)
        mask = mask.permute(1, 0, 2, 3).view(1, -1).type(torch.bool)
        instance_area = instance_area.permute(1, 0, 2, 3).view(1, -1)

        dotprod = (pred * targ).sum(dim=0, keepdims=True)
        dotprod = dotprod[mask]

        instance_area = instance_area[mask]

        angle_squared = (dotprod
                         .clamp(min=-1 + self.epsilon, max=1 - self.epsilon)
                         .acos()
                         .pow(2)
                         )

        weights = 1 / instance_area.sqrt()
        weighted_sum = (weights * angle_squared).sum()

        return weighted_sum / batch_size

 
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
