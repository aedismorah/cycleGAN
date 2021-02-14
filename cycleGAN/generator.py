import torch.nn as nn

def c7s1(in_channels, out_channels):
    return nn.Sequential(
        nn.ReflectionPad2d(2),
        nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(True)
    )

def dk(in_channels, out_channels):
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(True)
    )

def Rk(in_channels, out_channels):
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1),
        nn.ReflectionPad2d(1),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1),
        nn.InstanceNorm2d(out_channels)
    )

def uk(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(True)
    )

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.first = nn.Sequential(
            c7s1(3, 64),
            dk(64, 128),
            dk(128, 256)
        )

        self.R1 = Rk(256, 256)
        self.R2 = Rk(256, 256)
        self.R3 = Rk(256, 256)

        self.R4 = Rk(256, 256)
        self.R5 = Rk(256, 256)
        self.R6 = Rk(256, 256)

        self.R7 = Rk(256, 256)
        self.R8 = Rk(256, 256)
        self.R9 = Rk(256, 256)
        
        self.last = nn.Sequential(
            uk(256, 128),
            uk(128, 64),
            c7s1(64, 3),
            nn.Conv2d(3, 3, kernel_size=2, stride=1)
        )

        self.tanh=nn.Tanh()
        self.relu=nn.ReLU()

    def forward(self, input):
        x1 = self.first(input)

        x2 = self.relu(x1 + self.R1(x1))
        x3 = self.relu(x2 + self.R2(x2))
        x4 = self.relu(x3 + self.R3(x3))

        x5 = self.relu(x2 + x4 + self.R4(x4))
        x6 = self.relu(x5 + self.R5(x5))
        x7 = self.relu(x6 + self.R6(x6))

        x8 = self.relu(x3 + x5 + x7 + self.R7(x7))
        x9 = self.relu(x8 + self.R8(x8))
        x10 = self.relu(x2 + x6 + x9 + self.R9(x9))

        x = self.last(x10) + input

        return self.tanh(x)