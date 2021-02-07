import torch.nn as nn

def C64():
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=4, stride=2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
    )

def Ck(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
    )

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            C64(),
            Ck(64, 128),
            Ck(128, 256),
            Ck(256, 512),
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, input):
        return self.main(input)