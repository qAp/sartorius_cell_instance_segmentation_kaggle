
import torch.nn as nn

from seggit.data.config import WATERSHED_ENERGY_BINS


def net_params(outputChannels=len(WATERSHED_ENERGY_BINS) + 1, 
               wd=None, modelWeightPaths=None):
    params = {"depth/conv1_1": {"name": "depth/conv1_1", "shape": [5, 5, 2, 64], "std": None, "act": "relu", "reuse": False},
              "depth/conv1_2": {"name": "depth/conv1_2", "shape": [5, 5, 64, 128], "std": None, "act": "relu", "reuse": False},
              "depth/conv2_1": {"name": "depth/conv2_1", "shape": [5, 5, 128, 128], "std": None, "act": "relu", "reuse": False},
              "depth/conv2_2": {"name": "depth/conv2_2", "shape": [5, 5, 128, 128], "std": None, "act": "relu", "reuse": False},
              "depth/conv2_3": {"name": "depth/conv2_3", "shape": [5, 5, 128, 128], "std": None, "act": "relu", "reuse": False},
              "depth/conv2_4": {"name": "depth/conv2_4", "shape": [5, 5, 128, 128], "std": None, "act": "relu", "reuse": False},
              "depth/fcn1": {"name": "depth/fcn1", "shape": [1, 1, 128, 128], "std": None, "act": "relu", "reuse": False},
              "depth/fcn2": {"name": "depth/fcn2", "shape": [1, 1, 128, outputChannels], "std": None, "act": "relu", "reuse": False},
              "depth/upscore": {"name": "depth/upscore", "ksize": 8, "stride": 4, 'padding': 2, "outputChannels": outputChannels},
              }
    return params


class WatershedTransformNet(nn.Module):
    def __init__(self, data_config=None, args=None):
        super().__init__()

        self.args = vars(args) if args is not None else {}
        self.params = net_params()

        self.conv1_1 = self._conv_layer(self.params["depth/conv1_1"])
        self.conv1_2 = self._conv_layer(self.params["depth/conv1_2"])
        self.pool1 = self._avg_pool()

        self.conv2_1 = self._conv_layer(self.params["depth/conv2_1"])
        self.conv2_2 = self._conv_layer(self.params["depth/conv2_2"])
        self.conv2_3 = self._conv_layer(self.params["depth/conv2_3"])
        self.conv2_4 = self._conv_layer(self.params["depth/conv2_4"])
        self.pool2 = self._avg_pool()

        self.fcn1 = self._conv_layer(self.params["depth/fcn1"], dropout_p=.7)
        self.fcn2 = self._conv_layer(self.params["depth/fcn2"], dropout_p=.7)

        self.upscore_layer = self._upscore_layer(self.params["depth/upscore"]) # 4x

        # self.outputDataArgMax = tf.argmax(input=self.outputData, dimension=3)

    @staticmethod
    def add_argparse_args(parser):
        pass

    def forward(self, uvec):
        '''
        Args:
            uvec (N, 2, H, W)
        '''
        x = self.conv1_1(uvec)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv2_4(x)
        x = self.pool2(x)

        x = self.fcn1(x)
        x = self.fcn2(x)

        x = self.upscore_layer(x)

        return x

    def _conv_layer(self, params, dropout_p=None):
        kernel_height, kernel_width, in_channels, out_channels = params['shape']
        conv = nn.Conv2d(in_channels, out_channels,
                         kernel_size=(kernel_height, kernel_width),
                         stride=1, padding='same')

        layers = [conv]

        if params['act'] == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif params['act'] == 'lin':
            pass

        if dropout_p is not None:
            layers.append(nn.Dropout(p=dropout_p, inplace=False))

        return nn.Sequential(*layers)

    def _max_pool(self):
        return nn.MaxPool2d(kernel_size=2, stride=2)

    def _avg_pool(self, ):
        return nn.AvgPool2d(kernel_size=2, stride=2)

    def _upscore_layer(self, params):
        deconv = nn.ConvTranspose2d(in_channels=params['outputChannels'],
                                    out_channels=params['outputChannels'],
                                    kernel_size=params['ksize'],
                                    stride=params['stride'],
                                    padding=params['padding']
                                    )
        return deconv
