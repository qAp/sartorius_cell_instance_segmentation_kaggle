
import torch
import torch.nn as nn
import torchvision



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
