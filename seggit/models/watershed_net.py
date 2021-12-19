
import torch.nn as nn
import torch.nn.functional as F

import seggit
from seggit.models import DirectionNetMock
from seggit.models import WatershedTransformNet


def net_params():
    params_dn = seggit.models.direction_net.net_params()
    params_wtn = seggit.models.watershed_transform_net.net_params()

    params = {}
    params.update(params_dn)
    params.update(params_wtn)
    return params


class WatershedNet(nn.Module):
    def __init__(self, data_config=None, args=None):
        super().__init__()

        self.args = vars(args) if args is not None else {}

        self.dn = DirectionNetMock(data_config, args)
        self.wtn = WatershedTransformNet(data_config, args)

    @staticmethod
    def add_argparse_args(parser):
        DirectionNetMock.add_argparse_args(parser)
        WatershedTransformNet.add_argparse_args(parser)

    def forward(self, img, semg):
        '''
        Args:
            img (N, 3x1, H, W )
            semg (N, 1, H, W)
        '''
        uvec = self.dn(img, semg)        
        wngy = self.wtn(uvec)
        return wngy

        

