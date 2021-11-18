
import torch
import torch.nn as nn



class DirectionLoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, targ, mask, instance_area):
        '''
        Args:
            pred (torch.Tensor): Of size (height, width, 2)
                Predicted normalised gradient distance transform.
            targ (torch.Tensor): Of size (height, width, 2)
                Predicted normalised gradient distance transform.
            mask (torch.Tensor): Of size (height, width).  
                Semantic segmentation.
            instance_area (torch.Tensor): Of size (height, width).
                Area of instance in which pixel lies.
        '''
        pred = pred.permute(2, 0, 1).view(2, -1)
        targ = targ.permute(2, 0, 1).view(2, -1)
        mask = mask.type(torch.bool).view(1, -1)
        instance_area = instance_area.view(1, -1)

        dotprod = (pred * targ).sum(dim=0, keepdims=True)
        dotprod = dotprod[mask]

        instance_area = instance_area[mask]

        angle_squared = (dotprod
                         .clamp(min=-1 + self.epsilon, max=1 - self.epsilon)
                         .acos()
                         .pow(2)
                         )

        weighted_sum = (angle_squared * instance_area).sum()

        return weighted_sum
