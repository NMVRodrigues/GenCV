import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.initial = nn.Conv2d(in_channels=3, out_channels=32,  kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU()


        #TODO: Make this isto a function or loop
        self.d1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,  kernel_size=3, stride=1, padding=1),
            self.act,
            nn.MaxPool2d(2)
        )

        self.d2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,  kernel_size=3, stride=1, padding=1),
            self.act,
            nn.MaxPool2d(2)
        )

        self.d3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,  kernel_size=3, stride=1, padding=1),
            self.act,
            nn.MaxPool2d(2)
        )

        self.d4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,  kernel_size=3, stride=1, padding=1),
            self.act,
            nn.MaxPool2d(2)
        )



    def forward(self, x):
        x = self.initial(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.last = nn.Conv2d(in_channels=32, out_channels=3,  kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU()

        #TODO: Make this into a function or loop

        self.u1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256,  kernel_size=3, stride=1, padding=1),
            self.act,
            nn.Upsample(scale_factor=2)
        )

        self.u2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128,  kernel_size=3, stride=1, padding=1),
            self.act,
            nn.Upsample(scale_factor=2)
        )

        self.u3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64,  kernel_size=3, stride=1, padding=1),
            self.act,
            nn.Upsample(scale_factor=2)
        )

        self.u4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,  kernel_size=3, stride=1, padding=1),
            self.act,
            nn.Upsample(scale_factor=2)
        )


    def forward(self, x):
        x = self.u1(x)
        x = self.u2(x)
        x = self.u3(x)
        x = self.u4(x)
        x = self.last(x)
        return x