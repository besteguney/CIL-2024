import torch
import numpy as np
from blocks import Block
from torch import nn
from torchvision import models

class UNetVGG(nn.Module):
    def __init__(self, vgg_features, chs=(64,64,128,256,512,512)):
        super().__init__()
        self.enc1 = nn.Sequential(*vgg_features[0:5])   # Conv1 (2 conv layers + maxpool)
        self.enc2 = nn.Sequential(*vgg_features[5:10])  # Conv2 (2 conv layers + maxpool)
        self.enc3 = nn.Sequential(*vgg_features[10:17]) # Conv3 (3 conv layers + maxpool)
        self.enc4 = nn.Sequential(*vgg_features[17:24]) # Conv4 (3 conv layers + maxpool)
        self.enc5 = nn.Sequential(*vgg_features[24:31]) # Conv5 (3 conv layers + maxpool)
        self.encoders = [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]

        dec_chs = chs[::-1]  # decoder channels in the reverse order
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(in_ch, out_ch, 2, 2) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])])  # deconvolution
        self.dec_blocks = nn.ModuleList([Block(2*out_ch, out_ch) for out_ch in dec_chs[1:-1]])
        self.dec_blocks.append(Block(64,64))
        self.head = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())

    def forward(self, x):
        # encode
        enc_features = []
        output = x
        for i, encoder in enumerate(self.encoders):
          output = encoder(output)
          enc_features.append(output)

        # decode
        output = self.upconvs[0](enc_features[-1])
        output = torch.cat((output, enc_features[3]), dim=1)
        output = self.dec_blocks[0](output)

        for block, upconv, feature in zip(self.dec_blocks[1:-1], self.upconvs[1:-1], enc_features[::-1][2:]):
          output = upconv(output)
          output = torch.cat((output, feature), dim=1)
          output = block(output)

        output = self.upconvs[-1](output)
        output = self.dec_blocks[-1](output)
        return self.head(output)  # reduce to 1 channel

def get_unet_vgg_model(train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs, freezed_layers=24):
    vgg16 = models.vgg16(pretrained=True)
    
    for param in vgg16.features.parameters():
        param.requires_grad = False
    
    for param in vgg16.features[freezed_layers:].parameters():
        param.requires_grad = True
    
    vgg_features = list(vgg16.features.children())
    unet = UNetVGG(vgg_features)
    return unet
    