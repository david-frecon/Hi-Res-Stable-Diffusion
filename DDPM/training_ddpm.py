import torch

from NoiseScheduler import *
from UNet import *
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import to_device


BATCH_SIZE = 2**5
EPOCHS = 10
BETA = 0.0001
T_MAX = 200
BETAS = betas_schedule(BETA, T_MAX)
ALPHAS = alphas_schedule(BETAS)
ALPHAS_BAR = alphas_bar_schedule(ALPHAS)


stable_diff_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0,1),
    transforms.Lambda(lambda x: x * 2 - 1)
])
mnist_dataset = datasets.MNIST(root='./data', transform=stable_diff_transform, download=True)
mnist_7 = mnist_dataset.data[mnist_dataset.targets == 7]
dataloader_mnist = DataLoader(mnist_7, batch_size=BATCH_SIZE, shuffle=True)

unet = UNet(depth=4, time_emb_dim=32, color_channels=1)
unet = to_device(unet)

for epoch in range(EPOCHS):
    for batch_idx, batch in enumerate(dataloader_mnist):
        # generate diffente times
        batch = batch.to(unet.device)
        tensor_t = torch.randint(0, T_MAX, (BATCH_SIZE,))
        noise_batch = transforms.Lambda(lambda x: noise_sample(x, ALPHAS_BAR[tensor_t]))(batch)




