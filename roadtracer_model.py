import torch
import torch.nn as nn

# class Block(nn.Module):
#     # a repeating structure composed of two convolutional layers with batch normalization and ReLU activations
#     def __init__(self, in_ch, out_ch, activation='RELU'):
#         super().__init__()
#         self.activation = nn.ReLU() if activation == 'RELU' else nn.ELU()
#         # dims [batch, channels, distances, angles]
#         self.circ_pad = nn.CircularPad2d((1, 1, 0, 0))
#         self.zero_pad = nn.ConstantPad2d((0, 0, 1, 1), 0.0)
#         self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3) #, padding=1)
#         self.batch_norm = nn.BatchNorm2d(out_ch)
#         self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3) #, padding=1)

#     def forward(self, x):
#         x = torch.relu(self.conv1(self.zero_pad(self.circ_pad(x))))
#         x = self.batch_norm(x)
#         x = torch.relu(self.conv2(self.zero_pad(self.circ_pad(x))))
#         return x



# class RoadTracerUNet(nn.Module):
#     # UNet-like architecture for single class semantic segmentation.
#     def __init__(self, chs=(3,64,128,256,512,1024), activation='RELU'):
#         super().__init__()
#         enc_chs = chs  # number of channels in the encoder
#         dec_chs = chs[::-1][:-1]  # number of channels in the decoder

#         self.enc_blocks = nn.ModuleList([Block(in_ch, out_ch, activation) for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])])  # encoder blocks
#         self.dec_blocks = nn.ModuleList([Block(in_ch, out_ch, activation) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])])  # decoder blocks   
  
#         self.pool = nn.MaxPool2d(2)  # pooling layer (can be reused as it will not be trained)
#         self.upconvs = nn.ModuleList([nn.ConvTranspose2d(in_ch, out_ch, 2, 2) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])])  # deconvolution
#         self.activation = activation
    
#     def forward(self, x):
#         # encode
#         enc_features = []
#         for block in self.enc_blocks[:-1]:
#             x = block(x) # pass through the block
#             enc_features.append(x)  # save features for skip connections
#             x = self.pool(x)  # decrease resolution
#         x = self.enc_blocks[-1](x)
#         # decode
#         for block, upconv, feature in zip(self.dec_blocks, self.upconvs, enc_features[::-1]):
#             x = upconv(x)  # increase resolution
#             x = torch.cat([x, feature], dim=1)  # concatenate skip features
#             x = block(x)  # pass through the block
#         return x





# class RoadTracerModel(torch.nn.Module):
#     def __init__(self, distance_samples):
#         super().__init__()
#         self.unet = RoadTracerUNet()
#         self.pool = nn.MaxPool2d((2, 1))
#         self.channels = (64,128,256,512,512,512)
#         self.blocks = nn.ModuleList([Block(in_ch, out_ch) for in_ch, out_ch in zip(self.channels[:-1], self.channels[1:])])
#         self.head1 = torch.nn.Conv1d(1024 * (distance_samples // (2**(len(self.blocks)))), 256, 1)
#         self.head2 = torch.nn.Conv1d(256, 1, 1)

#     def forward(self, inputs):
#         # inputs = image [distances, angles, 3] 
#         # outputs = scores [angles] 
#         x = self.unet(inputs) # unet for global features
#         for block in self.blocks[:-1]:
#             x = block(x)
#             x = self.pool(x)
#         x = self.blocks[-1](x) # shape [batch, channels, dist, angles]
#         b, C, d, a = x.shape
#         x = x.reshape((b, C*d, a))
#         x = torch.relu(self.head1(x))
#         x = torch.sigmoid(self.head2(x)) # shape [batch, 1, angles]
#         return x[:, 0, :]


class Block2D(nn.Module):
    # a repeating structure composed of two convolutional layers with batch normalization and ReLU activations
    def __init__(self, in_ch, out_ch, activation='RELU'):
        super().__init__()
        self.activation = nn.ReLU() if activation == 'RELU' else nn.ELU()
        # dims [batch, channels, distances, angles]
        self.circ_pad = nn.CircularPad2d((1, 1, 0, 0))
        self.zero_pad = nn.ConstantPad2d((0, 0, 1, 1), 0.0)
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3) #, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3) #, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(self.zero_pad(self.circ_pad(x))))
        x = self.batch_norm(x)
        x = torch.relu(self.conv2(self.zero_pad(self.circ_pad(x))))
        return x


class Block1D(nn.Module):
    # a repeating structure composed of two convolutional layers with batch normalization and ReLU activations
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # dims [batch, channels, distances, angles]
        self.circ_pad = nn.CircularPad1d((1, 1))
        self.conv1 = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=3)
        self.batch_norm = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(in_channels=out_ch, out_channels=out_ch, kernel_size=3)

    def forward(self, x):
        x = torch.relu(self.conv1(self.circ_pad(x)))
        x = self.batch_norm(x)
        x = torch.relu(self.conv2(self.circ_pad(x)))
        return x


class RoadTracerUNet(nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self, chs=(1024, 1024, 1024, 1024, 1024, 1024, 1024)):
        super().__init__()
        enc_chs = chs  # number of channels in the encoder
        dec_chs = chs[::-1][:-1]  # number of channels in the decoder

        self.enc_blocks = nn.ModuleList([Block1D(in_ch, out_ch) for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])])  # encoder blocks
        self.dec_blocks = nn.ModuleList([Block1D(2*in_ch, out_ch) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])])  # decoder blocks   

        self.circ_pad = nn.CircularPad1d((1, 1))
        self.pool = nn.MaxPool1d(2)  # pooling layer (can be reused as it will not be trained)
        self.upconvs = nn.ModuleList([nn.ConvTranspose1d(in_ch, out_ch, 2, 2) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])])  # deconvolution
    
    def forward(self, inputs):
        x = inputs
        # encode
        enc_features = []
        for block in self.enc_blocks[:-1]:
            x = block(x) # pass through the block
            enc_features.append(x)  # save features for skip connections
            x = self.pool(x)  # decrease resolution
        x = self.enc_blocks[-1](x)
        # decode
        for block, upconv, feature in zip(self.dec_blocks, self.upconvs, enc_features[::-1]):
            x = upconv(x)  # increase resolution
            x = torch.cat([x, feature], dim=1)  # concatenate skip features
            x = block(x)  # pass through the block
        return x






# Idea - first downscale dist, then 1d unet for sides -> output
class RoadTracerModel(torch.nn.Module):
    def __init__(self, distance_samples):
        super().__init__()

        self.dist_channels = (3, 32, 64, 128, 256, 256)

        self.circ_pad = nn.CircularPad2d((1, 1, 0, 0))
        self.dist_downscale_blocks = nn.ModuleList([Block2D(in_ch, out_ch) for in_ch, out_ch in zip(self.dist_channels[:-1], self.dist_channels[1:])])
        self.dist_downscale_convs = nn.ModuleList([nn.Conv2d(ch, ch, (2, 3), (2, 1)) for ch in self.dist_channels[1:]])


        self.unet_1d = RoadTracerUNet()

        self.head1 = torch.nn.Conv1d(512 * (distance_samples // (2**(len(self.dist_channels)-1))), 256, 1)
        self.head2 = torch.nn.Conv1d(256, 1, 1)

    def forward(self, inputs):
        # inputs = image [batch, distances, angles, 3] 
        x = inputs
        for block, conv in zip(self.dist_downscale_blocks, self.dist_downscale_convs[:-1]):
            x = conv(self.circ_pad(block(x)))
        b, _, _, a = x.shape
        x = x.reshape((b, -1, a))
        x = self.unet_1d(x)
        x = torch.relu(self.head1(x))
        x = torch.sigmoid(self.head2(x)) # shape [batch, 1, angles]
        return x[:, 0, :]


        # outputs = scores [angles] 
        # x = self.unet(inputs) # unet for global features
        # for block in self.blocks[:-1]:
        #     x = block(x)
        #     x = self.pool(x)
        # x = self.blocks[-1](x) # shape [batch, channels, dist, angles]
        # b, C, d, a = x.shape
        # x = x.reshape((b, C*d, a))
        # x = torch.relu(self.head1(x))
        # x = torch.sigmoid(self.head2(x)) # shape [batch, 1, angles]
        # return x[:, 0, :]



        
