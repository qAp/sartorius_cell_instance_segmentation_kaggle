import os, sys
import torch.nn as nn
import torch.nn.functional as F

import seggit
from seggit.models import DirectionNetMock
from seggit.models import WatershedTransformNet
from seggit.lit_models import InstanceDirectionMockLitModel
from seggit.lit_models import WatershedEnergyLitModel


PRETRAINED_DN = None 
PRETRAINED_WTN = None


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

        self.pretrained_dn = self.args.get('pretrained_dn', PRETRAINED_DN)
        self.pretrained_wtn = self.args.get('pretrained_wtn', PRETRAINED_WTN)

        self.init_model_parameters()

    def init_model_parameters(self):
        dn = DirectionNetMock()
        wtn = WatershedTransformNet()

        if self.pretrained_dn is not None:
            assert os.path.exists(self.pretrained_dn)
            dn_litmodel = InstanceDirectionMockLitModel.load_from_checkpoint(
                checkpoint_path=self.pretrained_dn, model=dn)
            self.dn = dn_litmodel.model
            print(f'Loaded pretrained DN: {self.pretrained_dn}')
        else:
            self.dn = dn

        if self.pretrained_wtn is not None:
            assert os.path.exists(self.pretrained_wtn)
            wtn_litmodel = WatershedEnergyLitModel.load_from_checkpoint(
                checkpoint_path=self.pretrained_wtn, model=wtn)
            self.wtn = wtn_litmodel.model
            print(f'Loaded pretrained WTN: {self.pretrained_wtn}')
        else:
            self.wtn = wtn

    @staticmethod
    def add_argparse_args(parser):
        add = parser.add_argument
        DirectionNetMock.add_argparse_args(parser)
        WatershedTransformNet.add_argparse_args(parser)
        add('--pretrained_dn', type=str, default=PRETRAINED_DN)
        add('--pretrained_wtn', type=str, default=PRETRAINED_WTN)

    def forward(self, img, semg):
        '''
        Args:
            img (N, 3x1, H, W )
            semg (N, 1, H, W)
        '''
        uvec = self.dn(img, semg)        
        logits = self.wtn(uvec)
        return logits

        

