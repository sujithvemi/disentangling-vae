import torch
import torch.nn as nn

class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
        self.ds = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 2, padding=0),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        skip_op = self.ds(x)
        conv1_op = self.conv1(x)
        conv2_op = self.conv2(conv1_op)
        return nn.PReLU()(conv2_op + skip_op)
    
class DeconvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False):
        super().__init__()
        self.is_last = is_last
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, 3, 1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
        self.us = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 1, 2, padding=0, output_padding=1),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        skip_op = self.us(x)
        conv1_op = self.conv1(x)
        conv2_op = self.conv2(conv1_op)
        if not self.is_last:
            return nn.PReLU()(conv2_op + skip_op)
        else:
            return nn.Sigmoid()(conv2_op + skip_op)