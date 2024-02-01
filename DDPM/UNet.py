import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from utils import to_device


class TimeEmbedding(nn.Module):
    def __init__(self, dim=32):
        super(TimeEmbedding, self).__init__()
        self.dim = dim

    def forward(self, t):
        denominator = torch.pow(10000, torch.arange(0, self.dim, 2).float() / self.dim).unsqueeze(0)
        denominator = to_device(denominator)
        t = t.unsqueeze(1)
        sin = torch.sin(t / denominator)
        cos = torch.cos(t / denominator)
        emb = torch.stack((sin, cos), dim=2)
        emb = emb.view(t.shape[0], -1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, dim_time_embedding, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resize_time_emb = nn.Linear(dim_time_embedding, out_channels)
        self.upscale = in_channels > out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.upscale:
            self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, t, residual=None):
        if self.upscale:
            x = self.up_conv(x)
        if residual is not None:
            x = torch.cat((x, residual), dim=1)
        time_emb = F.relu(self.resize_time_emb(t))
        time_emb = time_emb[:, :, None, None]
        x = self.conv1(x)
        x = F.relu(x)
        x = x + time_emb
        x = self.conv2(x)
        x = F.relu(x)

        return x


class UNet(nn.Module):
    def __init__(self, depth=4, time_emb_dim=32, color_channels=1):
        super(UNet, self).__init__()
        self.channels = [2 ** (6 + i) for i in range(0, depth + 1)]  # 64 to 1024
        self.channels.insert(0, color_channels)
        self.reverse_channels = self.channels[::-1]
        self.depth = depth
        self.time_emb = TimeEmbedding(time_emb_dim)
        self.down_conv = nn.ModuleList(
            [ResidualBlock(time_emb_dim, self.channels[i], self.channels[i + 1]) for i in range(len(self.channels) - 2)])
        self.middle_conv = nn.Sequential(
            nn.Conv2d(self.channels[-2], self.channels[-1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels[-1], self.channels[-1], kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.up_conv = nn.ModuleList([ResidualBlock(time_emb_dim, self.reverse_channels[i], self.reverse_channels[i + 1]) for i in
                                      range(len(self.reverse_channels) - 2)])
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.out_conv = nn.Conv2d(self.channels[1], color_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, t):
        residuals = []
        for i, block in enumerate(self.down_conv):
            x = block(x, self.time_emb(t))
            residuals.append(x)
            x = self.max_pool(x)
        x = self.middle_conv(x)

        for i, block in enumerate(self.up_conv):
            x = block(x, self.time_emb(t), residuals[-i - 1])

        return self.out_conv(x)
