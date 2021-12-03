
import torch.nn as nn

from seggit.data.config import WATERSHED_ENERGY_BINS



class WatershedTransformNet(nn.Module):
    def __init__(self, n_energy=len(WATERSHED_ENERGY_BINS) + 1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=256,
                      kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.fcn = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=1, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=n_energy,
                      kernel_size=1, stride=1, padding=1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.fcn(x)

        x = nn.Upsample(input_size)(x)

        return x
