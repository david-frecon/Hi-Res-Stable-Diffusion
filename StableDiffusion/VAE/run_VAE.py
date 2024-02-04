import torch
import matplotlib.pyplot as plt

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from StableDiffusion.utils import to_device, get_device_name
from VAE import VAE


MODEL_NAME = "small_vae.pt"
INDEXES_TO_PLOT = [0, 1, 2, 3, 4]


dataset = datasets.ImageFolder(root='../data/data_for_fashion_clip/out/', transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)
]))

net = to_device(VAE())
net.load_state_dict(torch.load(f"../models/{MODEL_NAME}", map_location=get_device_name()))
net.eval()
net.requires_grad_(False)

plt.figure(figsize=(10, 8))
for i, index in enumerate(INDEXES_TO_PLOT):
    x, _ = dataset[index]

    reconstruction, _, _ = net(to_device(x.unsqueeze(0)))
    reconstruction = reconstruction.clip(-1, 1)
    reconstruction = (reconstruction + 1) / 2
    reconstruction = reconstruction * 255
    reconstruction = reconstruction.cpu().numpy().astype(int)

    ax = plt.subplot(len(INDEXES_TO_PLOT), 2, i * 2 + 1)
    ax.axis("off")
    ax.set_title("Original")
    xtmp = x.cpu().numpy().transpose(1, 2, 0)
    xtmp = (xtmp + 1) / 2
    ax.imshow(xtmp)
    ax = plt.subplot(len(INDEXES_TO_PLOT), 2, i * 2 + 2)
    ax.axis("off")
    ax.set_title("Reconstruction")
    ax.imshow(reconstruction[0].transpose(1, 2, 0))

plt.show()
