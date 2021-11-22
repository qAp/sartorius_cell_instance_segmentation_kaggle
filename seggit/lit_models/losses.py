
import torch
import torch.nn as nn



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

        # weighted_sum = (angle_squared * instance_area).sum()

        # return weighted_sum / batch_size

        weighted_mean = (angle_squared * instance_area).mean()
        return weighted_mean

 
