import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.inchannels = in_channels
        self.outchannels = out_channels
        self.upscale = in_channels > out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.upscale:
            self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, residual=None):
        if self.upscale:
            x = self.up_conv(x)
        if residual is not None:
            x = torch.cat((x, residual), dim=1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class UNet(nn.Module):
    def __init__(self, depth=4, color_channels=1):
        super(UNet, self).__init__()
        self.channels = [2 ** (6 + i) for i in range(0, depth + 1)]  # 64 to 1024
        self.channels.insert(0, color_channels)
        self.reverse_channels = self.channels[::-1]
        self.depth = depth
        self.down_conv = nn.ModuleList([ResidualBlock(self.channels[i], self.channels[i+1]) for i in range(len(self.channels) - 2)])
        self.middle_conv = nn.Sequential(
            nn.Conv2d(self.channels[-2], self.channels[-1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels[-1], self.channels[-1], kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.up_conv = nn.ModuleList([ResidualBlock(self.reverse_channels[i], self.reverse_channels[i+1]) for i in range(len(self.reverse_channels) - 2)])
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.out_conv = nn.Conv2d(self.channels[1], color_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residuals = []
        for i, block in enumerate(self.down_conv):
            x = block(x)
            residuals.append(x)
            x = self.max_pool(x)
        x = self.middle_conv(x)

        for i, block in enumerate(self.up_conv):
            x = block(x, residuals[-i-1])

        return self.out_conv(x)


if __name__ == '__main__':
    model = UNet()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    summary(model, (1, 128, 128))

