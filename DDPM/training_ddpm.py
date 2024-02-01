import torch

from NoiseScheduler import *
from UNet import *
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import to_device, test_chain
from datetime import datetime

BATCH_SIZE = 2**5
EPOCHS = 500
LR = 0.00001
BETA = 0.0001
T_MAX = 400
BETAS = betas_schedule(BETA, T_MAX)
ALPHAS = alphas_schedule(BETAS)
ALPHAS_BAR = alphas_bar_schedule(ALPHAS)


stable_diff_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(0, 1)
    transforms.Lambda(lambda x: x * 2 - 1)
])
# mnist_dataset = datasets.MNIST(root='./data', transform=stable_diff_transform, download=True)
cifar_dataset = datasets.CIFAR10(root='./data', transform=stable_diff_transform, download=True)
# mnist_7 = [img for img, label in mnist_dataset if label == 7]
cifar_7 = [img for img, label in cifar_dataset if label == 7]
print("cifar_7", len(cifar_7))
# print(len(mnist_7))
# dataloader_mnist = DataLoader(mnist_7, batch_size=BATCH_SIZE, shuffle=True)
dataloader_cifar = DataLoader(cifar_7, batch_size=BATCH_SIZE, shuffle=True)
unet = UNet(depth=4, time_emb_dim=32, color_channels=3)
unet = to_device(unet)

optimizer = torch.optim.Adam(unet.parameters(), lr=LR)

for epoch in range(EPOCHS):
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}")
    for batch_idx, batch in enumerate(dataloader_cifar):
        if len(batch) != BATCH_SIZE:
            continue

        tensor_t = torch.randint(0, T_MAX, (BATCH_SIZE,))
        tensor_t_float = to_device(tensor_t.float())
        noised_batch, noises = noise_batch(batch, ALPHAS_BAR, tensor_t)
        noises = to_device(noises)
        noised_batch = to_device(noised_batch)

        optimizer.zero_grad()
        output = unet(noised_batch, tensor_t_float)
        loss = F.mse_loss(output, noises)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 50 == 0:
            if batch_idx % 100 == 0:
                print(f"Loss: {loss.item()}")
                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(denormalize_img(batch[0].permute(1, 2, 0).detach().cpu().squeeze()))
                ax[1].imshow(denormalize_img(noises[0].permute(1, 2, 0).detach().cpu().squeeze()))
                ax[2].imshow(denormalize_img(output[0].permute(1, 2, 0).detach().cpu().squeeze()))
                plt.show()


torch.save(unet, f"models/unet_{T_MAX}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pth")
test_chain(unet, BETA, T_MAX)
