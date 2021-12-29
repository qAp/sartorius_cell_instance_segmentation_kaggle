
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



class DirectionNetBG(nn.Module):
    def __init__(self, data_config=None, args=None):
        super().__init__()

        self.args = vars(args) if args is not None else {}

        self.params = net_params() 

        self.pretrained_vgg16 = self.args.get('pretrained_vgg16', False)

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

    @staticmethod
    def add_argparse_args(parser):
        add = parser.add_argument
        add('--pretrained_vgg16', action='store_true', default=False)

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

    def forward(self, img, semseg):
        '''
        Args:
            img (N, 3x1, H, W )
            semseg (N, 3, H, W)

        '''
        ss = semseg.sum(dim=1, keepdim=True)    # ss (N, 1, H, W)

        nothing = torch.zeros_like(semseg[:,[0],:,:])
        ssmask = torch.cat(
            [
                semseg[:,[1, 2],:,:].sum(dim=1, keepdim=True), 
                nothing, 
                semseg[:,[0],:,:]
            ], 
            dim=1)
        ssmask = ssmask.argmax(dim=1, keepdim=True)
        ssmask = ssmask.type(torch.float32)
        ssmask = 32 * (ssmask - 1)             # ssmssk (N, 1, H, W)

        x = torch.cat([ss * img, ssmask], dim=1)

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

        x = ss * x

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
