import os
import sys
import torch

from torch import nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.parts import Encoder, Decoder, VAE

data = torch.randn(1, 3, 32, 32)

encoder = Encoder(in_channels=3)
decoder = Decoder(out_channels=3)

out_e = encoder(data)
out_d = decoder(out_e)

print(out_e.shape)
print(out_d.shape)

vae = VAE(input_size=32, in_channels=3)
out_vae = vae(data)

print(out_vae.shape)