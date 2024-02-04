import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from StableDiffusion.utils import to_device
from VAE import VAE, reconstruction, kl_div_gauss

MODEL_NAME = "vae.pt"
EPOCHS = 100
BETA = 1.0
LATENT_DIM = 16
LR = 0.001

dataset = datasets.ImageFolder(root='../data/data_for_fashion_clip/out/', transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)
]))
loader = DataLoader(dataset, batch_size=128, shuffle=True)

net = to_device(VAE(latent_dim=LATENT_DIM))
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

for epoch in range(EPOCHS):
    _recon, _kl = 0., 0.
    counter = 0

    for x, _ in loader:
        x = to_device(x)

        optimizer.zero_grad()
        xhat, mean, log_var = net(x)

        recon_loss = reconstruction(xhat, x)
        kl_loss = kl_div_gauss(log_var, mean)

        loss = recon_loss + BETA * kl_loss
        loss.backward()
        optimizer.step()

        _recon += recon_loss.item()
        _kl += kl_loss.item()
        counter += 1

    _recon = round(_recon / counter, 5)
    _kl = round(_kl / counter, 5)

    print(f"Epoch {epoch}; recon: {_recon}; kl: {_kl}")

torch.save(net.state_dict(), f"../models/{MODEL_NAME}")
