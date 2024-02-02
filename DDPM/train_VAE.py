import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from VAE import VAE, reconstruction, kl_div_gauss

dataset = datasets.ImageFolder(root='data/data_for_fashion_clip/out/', transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)
]))

loader = DataLoader(dataset, batch_size=128, shuffle=True)

epochs = 100
beta = 1.0

net = VAE().cuda()
optimizer = torch.optim.Adam(net.parameters())

for epoch in range(epochs):
    _recon, _kl = 0., 0.
    c = 0

    for x, _ in loader:
        x = x.cuda()

        optimizer.zero_grad()
        xhat, mean, log_var = net(x)

        recon_loss = reconstruction(xhat, x)
        kl_loss = kl_div_gauss(log_var, mean)

        loss = recon_loss + beta * kl_loss
        loss.backward()
        optimizer.step()

        _recon += recon_loss.item()
        _kl += kl_loss.item()
        c += 1

    _recon = round(_recon / c, 5)
    _kl = round(_kl / c, 5)

    print(f"Epoch {epoch}; recon: {_recon}; kl: {_kl}")

torch.save(net.state_dict(), f"models/vae.pt")
