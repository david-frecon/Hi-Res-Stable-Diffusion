import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()

        self.latent_dim = latent_dim

        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 16 * 64, 1024),
            nn.ReLU(inplace=True)
        )

        self.fc_mean = nn.Linear(1024, latent_dim)
        self.fc_log_var = nn.Linear(1024, latent_dim)

    def forward(self, x):
        x = self.enc(x)
        x = x.view(len(x), -1)
        x = self.fc(x)

        mean = self.fc_mean(x)
        # Our encoder computes implicitly the log var
        # instead of the var, in order to avoid using
        # a numerically instable logarithm afterwards.
        log_var = self.fc_log_var(x)

        z = mean + torch.randn_like(mean) * torch.exp(log_var)

        return z, mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True)
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(1024, 64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=8, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=8, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=8, stride=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=8, stride=3, padding=15)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(len(x), 1024, 1, 1)
        x = self.dec(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim=16**2):
        super().__init__()

        self.enc = Encoder(latent_dim)
        self.dec = Decoder(latent_dim)

    def forward(self, x):
        z, mean, log_var = self.enc(x)
        xhat = self.dec(z)
        return xhat, mean, log_var


def reconstruction(xhat, x):
    return torch.pow(xhat - x, 2).sum(dim=(1, 2, 3)).mean()


def kl_div_gauss(log_var, mean):
    kl_loss = -0.5 * (1 + log_var - torch.pow(mean, 2) - torch.exp(log_var))
    kl_loss = kl_loss.sum(dim=1).mean()
    return kl_loss
