
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



def net_params():
    fuseChannels = 256
    outputChannels = 2
    params = {"direction/conv1_1": {"name": "direction/conv1_1", "shape": [3, 3, 4, 64], "std": None, "act": "relu"},
              "direction/conv1_2": {"name": "direction/conv1_2", "shape": [3, 3, 64, 64], "std": None, "act": "relu"},
              "direction/conv2_1": {"name": "direction/conv2_1", "shape": [3, 3, 64, 128], "std": None, "act": "relu"},
              "direction/conv2_2": {"name": "direction/conv2_2", "shape": [3, 3, 128, 128], "std": None, "act": "relu"},
              "direction/conv3_1": {"name": "direction/conv3_1", "shape": [3, 3, 128, 256], "std": None, "act": "relu"},
              "direction/conv3_2": {"name": "direction/conv3_2", "shape": [3, 3, 256, 256], "std": None, "act": "relu"},
              "direction/conv3_3": {"name": "direction/conv3_3", "shape": [3, 3, 256, 256], "std": None, "act": "relu"},
              "direction/conv4_1": {"name": "direction/conv4_1", "shape": [3, 3, 256, 512], "std": None, "act": "relu"},
              "direction/conv4_2": {"name": "direction/conv4_2", "shape": [3, 3, 512, 512], "std": None, "act": "relu"},
              "direction/conv4_3": {"name": "direction/conv4_3", "shape": [3, 3, 512, 512], "std": None, "act": "relu"},
              "direction/conv5_1": {"name": "direction/conv5_1", "shape": [3, 3, 512, 512], "std": None, "act": "relu"},
              "direction/conv5_2": {"name": "direction/conv5_2", "shape": [3, 3, 512, 512], "std": None, "act": "relu"},
              "direction/conv5_3": {"name": "direction/conv5_3", "shape": [3, 3, 512, 512], "std": None, "act": "relu"},
              "direction/fcn5_1": {"name": "direction/fcn5_1", "shape": [5, 5, 512, 512], "std": None, "act": "relu"},
              "direction/fcn5_2": {"name": "direction/fcn5_2", "shape": [1, 1, 512, 512], "std": None, "act": "relu"},
              "direction/fcn5_3": {"name": "direction/fcn5_3", "shape": [1, 1, 512, fuseChannels], "std": 1e-2, "act": "relu"},
              "direction/upscore5_3": {"name": "direction/upscore5_3", "ksize": 8, "stride": 4, 'padding': 2, "outputChannels": fuseChannels},
              "direction/fcn4_1": {"name": "direction/fcn4_1", "shape": [5, 5, 512, 512], "std": None, "act": "relu"},
              "direction/fcn4_2": {"name": "direction/fcn4_2", "shape": [1, 1, 512, 512], "std": None, "act": "relu"},
              "direction/fcn4_3": {"name": "direction/fcn4_3", "shape": [1, 1, 512, fuseChannels], "std": 1e-3, "act": "relu"},
              "direction/upscore4_3": {"name": "direction/upscore4_3", "ksize": 4, "stride": 2, 'padding': 1, "outputChannels": fuseChannels},
              "direction/fcn3_1": {"name": "direction/fcn3_1", "shape": [5, 5, 256, 256], "std": None, "act": "relu"},
              "direction/fcn3_2": {"name": "direction/fcn3_2", "shape": [1, 1, 256, 256], "std": None, "act": "relu"},
              "direction/fcn3_3": {"name": "direction/fcn3_3", "shape": [1, 1, 256, fuseChannels], "std": 1e-4, "act": "relu"},
              "direction/fuse3_1": {"name": "direction/fuse_1", "shape": [1, 1, fuseChannels*3, 512], "std": None, "act": "relu"},
              "direction/fuse3_2": {"name": "direction/fuse_2", "shape": [1, 1, 512, 512], "std": None, "act": "relu"},
              "direction/fuse3_3": {"name": "direction/fuse_3", "shape": [1, 1, 512, outputChannels], "std": None, "act": "lin"},
              "direction/upscore3_1": {"name": "direction/upscore3_1", "ksize": 8, "stride": 4, 'padding': 2, "outputChannels": outputChannels}}
    return params


def max_to_avgpool(m):
    assert isinstance(m, nn.MaxPool2d)
    return nn.AvgPool2d(kernel_size=m.kernel_size,
                        stride=m.stride,
                        padding=m.padding,
                        ceil_mode=m.ceil_mode)


class FCN(nn.Module):
    def __init__(self, channels=32):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=1, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=1, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class Fuse(nn.Module):
    def __init__(self, in_channels=1280, out_channels=2):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=1, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=1, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
         
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DirectionNet(nn.Module):
    def __init__(self):
        super().__init__()

        m = torchvision.models.vgg16(pretrained=True).features

        self.conv1 = m[:4]
        self.pool1 = m[4]
        self.conv2 = m[5:9]
        self.pool2 = m[9]
        self.conv3 = m[10:16]
        self.pool3 = m[16]
        self.conv4 = m[17:23]
        self.pool4 = m[23]
        self.conv5 = m[24:30]

        self.pool3 = max_to_avgpool(self.pool3)
        self.pool4 = max_to_avgpool(self.pool4)

        self.fcn3 = FCN(channels=256)
        self.fcn4 = FCN(channels=512)
        nn.init.xavier_normal_(self.fcn4.conv3[0].weight, gain=3)
        self.fcn5 = FCN(channels=512)
        nn.init.xavier_normal_(self.fcn5.conv3[0].weight, gain=8)

        self.fuse3 = Fuse(in_channels=1280, out_channels=2)

    def forward(self, x):
        input_size = x.shape[-2:]

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        x3 = self.fcn3(x)

        x = self.pool3(x)
        x = self.conv4(x)

        x4 = self.fcn4(x)
        x4 = nn.Upsample(x3.shape[-2:])(x4)

        x = self.pool4(x)
        x = self.conv5(x)

        x5 = self.fcn5(x)
        x5 = nn.Upsample(x3.shape[-2:])(x5)

        x = torch.cat([x3, x4, x5], dim=1)

        x = self.fuse3(x)

        x = nn.Upsample(input_size)(x)

        return x


class DirectionNetMock(nn.Module):
    def __init__(self, params=None, pretrained_vgg16=False):
        super().__init__()
        self.params = net_params() if params is None else params
        self.pretrained_vgg16 = pretrained_vgg16

        self.conv1_1 = self._conv_layer(self.params['direction/conv1_1'])
        self.conv1_2 = self._conv_layer(self.params['direction/conv1_2'])
        self.pool1 = self._max_pool()

        self.conv2_1 = self._conv_layer(self.params['direction/conv2_1'])
        self.conv2_2 = self._conv_layer(self.params['direction/conv2_2'])
        self.pool2 = self._max_pool()

        self.conv3_1 = self._conv_layer(self.params['direction/conv3_1'])
        self.conv3_2 = self._conv_layer(self.params['direction/conv3_2'])
        self.conv3_3 = self._conv_layer(self.params['direction/conv3_3'])
        self.pool3 = self._avg_pool()

        self.conv4_1 = self._conv_layer(self.params['direction/conv4_1'])
        self.conv4_2 = self._conv_layer(self.params['direction/conv4_2'])
        self.conv4_3 = self._conv_layer(self.params['direction/conv4_3'])
        self.pool4 = self._avg_pool()

        self.conv5_1 = self._conv_layer(self.params['direction/conv5_1'])
        self.conv5_2 = self._conv_layer(self.params['direction/conv5_2'])
        self.conv5_3 = self._conv_layer(self.params['direction/conv5_3'])

        self.fcn5_1 = self._conv_layer(self.params['direction/fcn5_1'])
        self.fcn5_2 = self._conv_layer(self.params['direction/fcn5_2'])
        self.fcn5_3 = self._conv_layer(self.params['direction/fcn5_3'])

        self.fcn4_1 = self._conv_layer(self.params['direction/fcn4_1'])
        self.fcn4_2 = self._conv_layer(self.params['direction/fcn4_2'])
        self.fcn4_3 = self._conv_layer(self.params['direction/fcn4_3'])

        self.fcn3_1 = self._conv_layer(self.params['direction/fcn3_1'])
        self.fcn3_2 = self._conv_layer(self.params['direction/fcn3_2'])
        self.fcn3_3 = self._conv_layer(self.params['direction/fcn3_3'])

        self.upscore5_3 = self._upscore_layer(
            self.params['direction/upscore5_3'])  # 4x
        self.upscore4_3 = self._upscore_layer(
            self.params['direction/upscore4_3'])  # 2x

        self.fuse3_1 = self._conv_layer(self.params['direction/fuse3_1'])
        self.fuse3_2 = self._conv_layer(self.params['direction/fuse3_2'])
        self.fuse3_3 = self._conv_layer(self.params['direction/fuse3_3'])

        self.upscore_layer = self._upscore_layer(
            self.params['direction/upscore3_1'])  # 4x

        self.init_model_parameters()

    def init_model_parameters(self):
        if self.pretrained_vgg16:
            vgg16 = torchvision.models.vgg16(pretrained=True).features

            for ldn, lvgg in self.vgg16_mapping().items():
                conv_dn = getattr(self, ldn)[0]
                conv_vgg = vgg16[lvgg]
                assert conv_dn.weight.shape == conv_vgg.weight.shape
                assert conv_dn.bias.shape == conv_vgg.bias.shape

                conv_dn.weight.data = conv_vgg.weight.data
                conv_dn.bias.data = conv_vgg.bias.data

    def vgg16_mapping(self):
        return {'conv1_2': 2,
                'conv2_1': 5, 'conv2_2': 7,
                'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14,
                'conv4_1': 17, 'conv4_2': 19, 'conv4_3': 21,
                'conv5_1': 24, 'conv5_2': 26, 'conv5_3': 28}   

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        conv3_3 = self.conv3_3(x)
        x = self.pool3(conv3_3)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        conv4_3 = self.conv4_3(x)
        x = self.pool4(conv4_3)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        conv5_3 = self.conv5_3(x)

        fcn5_1 = self.fcn5_1(conv5_3)
        fcn5_2 = self.fcn5_2(fcn5_1)
        fcn5_3 = self.fcn5_3(fcn5_2)

        fcn4_1 = self.fcn4_1(conv4_3)
        fcn4_2 = self.fcn4_2(fcn4_1)
        fcn4_3 = self.fcn4_3(fcn4_2)

        fcn3_1 = self.fcn3_1(conv3_3)
        fcn3_2 = self.fcn3_2(fcn3_1)
        fcn3_3 = self.fcn3_3(fcn3_2)

        upscore5_3 = self.upscore5_3(fcn5_3)
        upscore4_3 = self.upscore4_3(fcn4_3)

        fuse3 = torch.cat([fcn3_3, upscore4_3, upscore5_3], dim=1)

        x = self.fuse3_1(fuse3)
        x = self.fuse3_2(x)
        x = self.fuse3_3(x)

        x = self.upscore_layer(x)
        # TODO: Gate with `ss`

        x = F.normalize(x, dim=1)
        return x

    def _conv_layer(self, params):
        kernel_height, kernel_width, in_channels, out_channels = params['shape']
        conv = nn.Conv2d(in_channels, out_channels,
                         kernel_size=(kernel_height, kernel_width),
                         stride=1, padding='same')

        if params['act'] == 'relu':
            return nn.Sequential(conv, nn.ReLU(inplace=True))
        elif params['act'] == 'lin':
            return conv

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
