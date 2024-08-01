import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

KERNEL_SIZE = 3

class RoadTracerModel(nn.Module):
    def __init__(self, patch_size, input_channels=4, num_angles=64):
        super(RoadTracerModel, self).__init__()
        self.num_angles = num_angles
        self.layer1 = self._conv_layer(input_channels, 64, stride=2)
        self.layer2 = self._conv_layer(64, 64, stride=1)
        self.layer3 = self._conv_layer(64, 128, stride=2)
        self.layer4 = self._conv_layer(128, 128, stride=1)
        self.layer5 = self._conv_layer(128, 256, stride=2)
        self.layer6 = self._conv_layer(256, 256, stride=1)
        self.layer7 = self._conv_layer(256, 512, stride=2)
        self.layer8 = self._conv_layer(512, 512, stride=1)


        self.upconv1 = self._upconv_layer(256, 128)
        self.upconv2 = self._upconv_layer(128, 64)
        self.detect_pre_outputs = nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2, padding=0)

        last_layer_size = patch_size // 16
        self.fc_action = nn.Sequential(
            nn.Linear(512 * last_layer_size * last_layer_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

        self.fc_angle = nn.Sequential(
            nn.Linear(512 * last_layer_size * last_layer_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_angles)
        )

    def _conv_layer(self, in_channels, out_channels, stride, activation='relu'):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=KERNEL_SIZE, stride=stride, padding=KERNEL_SIZE//2))
        layers.append(nn.BatchNorm2d(out_channels))
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def _upconv_layer(self, in_channels, out_channels):
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        layer5_out = self.layer5(x)
        x = self.layer6(layer5_out)
        x = self.layer7(x)
        layer8_out = self.layer8(x)

        x = self.upconv1(layer5_out)
        x = self.upconv2(x)           
        detect_outputs = self.detect_pre_outputs(x)  # shape: [BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE]
        
        # flatten the output for FC layers
        # x_flat = layer8_out.view(x.size(0), -1)
        x_flat = layer8_out.reshape(x.size(0), -1)

        action_outputs = self.fc_action(x_flat)  # shape: [BATCH_SIZE, 2]
        angle_outputs = torch.sigmoid(self.fc_angle(x_flat))  # shape: [BATCH_SIZE, num_angles]

        return action_outputs, angle_outputs, detect_outputs


# class RoadTracerModel(nn.Module):
#     def __init__(self, patch_size, input_channels=4, num_angles=64):
#         super(RoadTracerModel, self).__init__()
#         self.num_angles = num_angles
        
#         self.layer1 = self._conv_layer(input_channels, 32, stride=2, kernel_size=5)
#         self.layer2 = self._conv_layer(32, 64, stride=2, kernel_size=5)
#         self.layer3 = self._conv_layer(64, 128, stride=2, kernel_size=3)
#         self.layer4 = self._conv_layer(128, 256, stride=2, kernel_size=3)
#         self.layer5 = self._conv_layer(256, 512, stride=2, kernel_size=3)

#         self.upconv3 = self._upconv_layer(128, 64, stride=2, kernel_size=3)
#         self.upconv2 = self._upconv_layer(64 + 64, 32, stride=2, kernel_size=5)
#         self.upconv1 = self._upconv_layer(32 + 32, 16, stride=2, kernel_size=5)
#         self.detect_pre_outputs = nn.ConvTranspose2d(16, 1, kernel_size=5, stride=1, padding=2)

#         last_layer_size = patch_size // 32
#         self.fc_action = nn.Sequential(
#             nn.Linear(512 * last_layer_size * last_layer_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, 2)
#         )

#         self.fc_angle = nn.Sequential(
#             nn.Linear(512 * last_layer_size * last_layer_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_angles)
#         )

#     def _conv_layer(self, in_channels, out_channels, stride, kernel_size=3, dilation=1, activation='relu'):
#         layers = [
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size // 2) * dilation, dilation=dilation),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True) if activation == 'relu' else nn.Sigmoid()
#         ]
#         return nn.Sequential(*layers)

#     def _upconv_layer(self, in_channels, out_channels, stride, kernel_size=3):
#         layers = [
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, output_padding=stride - 1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         ]
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x1 = self.layer1(x)
#         x2 = self.layer2(x1)
#         x3 = self.layer3(x2)
#         x4 = self.layer4(x3)
#         layer5_out = self.layer5(x4)

#         x = self.upconv3(x3)
#         x = self.upconv2(torch.cat([x, x2], dim=1))
#         x = self.upconv1(torch.cat([x, x1], dim=1))
#         detect_outputs = self.detect_pre_outputs(x)  # Shape: [BATCH_SIZE, 1, patch_size, patch_size]

#         # flatten the output for FC layers
#         x_flat = layer5_out.reshape(layer5_out.size(0), -1)

#         action_outputs = self.fc_action(x_flat)  # Shape: [BATCH_SIZE, 2]
#         angle_outputs = torch.sigmoid(self.fc_angle(x_flat))  # Shape: [BATCH_SIZE, num_angles]

#         return action_outputs, angle_outputs, detect_outputs
