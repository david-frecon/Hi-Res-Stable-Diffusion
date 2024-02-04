import torch
import torch.nn as nn
import torch.nn.functional as F

import StableDiffusion.UNet.UNet as un


class ResidualBlock(un.ResidualBlock):
    def __init__(self, dim_time_embedding, dim_text_embedding, in_channels, out_channels):
        super(ResidualBlock, self).__init__(dim_time_embedding, in_channels, out_channels)
        self.resize_text_emb = nn.Linear(dim_text_embedding, out_channels)

    def forward(self, x, t, text_emb, residual=None):
        if self.upscale:
            x = self.up_conv(x)
        if residual is not None:
            x = torch.cat((x, residual), dim=1)
        time_emb = F.relu(self.resize_time_emb(t))
        time_emb = time_emb[:, :, None, None]
        text_emb = F.relu(self.resize_text_emb(text_emb))
        text_emb = text_emb[:, :, None, None]
        x = self.conv1(x)
        x = self.gnorm1(F.relu(x))
        x = x + time_emb + text_emb
        x = self.conv2(x)
        x = self.gnorm2(F.relu(x))

        return x


class UNetText(un.UNet):
    def __init__(self, depth=4, time_emb_dim=32, text_emb_dim=512, color_channels=1):
        self.text_emb_dim = text_emb_dim
        super(UNetText, self).__init__(depth, time_emb_dim, color_channels)

    def init_blocks(self):
        self.down_conv = nn.ModuleList(
            [ResidualBlock(self.time_emb_dim, self.text_emb_dim, self.channels[i],
                           self.channels[i + 1]) for i in range(len(self.channels) - 2)])

        self.middle_conv = nn.Sequential(
            nn.Conv2d(self.channels[-2], self.channels[-1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels[-1], self.channels[-1], kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.up_conv = nn.ModuleList([
            ResidualBlock(self.time_emb_dim, self.text_emb_dim, self.reverse_channels[i],
                          self.reverse_channels[i + 1]) for i in range(len(self.reverse_channels) - 2)])

    def forward(self, x, t, text_emb):
        residuals = []
        for i, block in enumerate(self.down_conv):
            x = block(x, self.time_emb(t), text_emb)
            residuals.append(x)
            x = self.max_pool(x)
            x = self.drop(x)
        x = self.middle_conv(x)

        for i, block in enumerate(self.up_conv):
            x = block(x, self.time_emb(t), text_emb, residuals[-i - 1])
            x = self.drop(x)

        return self.out_conv(x)
