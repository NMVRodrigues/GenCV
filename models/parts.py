import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_sizes=[32,64,128,256,512]):	
        super().__init__()

        self.initial = nn.Conv2d(in_channels=in_channels, out_channels=latent_sizes[0],  kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU()

        self.downs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels=latent_sizes[i], out_channels=latent_sizes[i]*2,  kernel_size=3, stride=1, padding=1),
                    self.act,
                    nn.MaxPool2d(2)
                )
                for i in range(len(latent_sizes)-1)
            ]

        )	



    def forward(self, x):
        x = self.initial(x)
        for d in self.downs:
            x = d(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels=1, latent_sizes=[512,256,128,64,32]):
        super().__init__()
        self.last = nn.Conv2d(in_channels=latent_sizes[-1], out_channels=out_channels,  kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU()

        self.ups = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels=latent_sizes[i], out_channels=latent_sizes[i]//2,  kernel_size=3, stride=1, padding=1),
                    self.act,
                    nn.Upsample(scale_factor=2)
                )
                for i in range(len(latent_sizes)-1)
            ]

        )


    def forward(self, x):
        for u in self.ups:
            x = u(x)
        x = self.last(x)
        return x
    

class VAE(nn.Module):
    def __init__(self, in_channels=1, latent_sizes=[32,64,128,256,512]):	
        super().__init__()

        self.encoder = Encoder(in_channels=in_channels, latent_sizes=latent_sizes)
        self.decoder = Decoder(out_channels=in_channels, latent_sizes=latent_sizes)



    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x