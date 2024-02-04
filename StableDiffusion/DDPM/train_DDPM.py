import math

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from rich.progress import SpinnerColumn
from rich.progress import Progress
import time

from StableDiffusion.NoiseScheduler.NoiseScheduler import *
from StableDiffusion.UNet.UNet import UNet
from StableDiffusion.utils import to_device, test_DDPM_chain

BATCH_SIZE = 2**6
EPOCHS = 200
LR = 0.0001
BETA = 0.0001
T_MAX = 1000
BETAS = betas_schedule(BETA, T_MAX)
ALPHAS = alphas_schedule(BETAS)
ALPHAS_BAR = alphas_bar_schedule(ALPHAS)


stable_diff_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)
])
cifar_dataset = datasets.CIFAR10(root='./data', transform=stable_diff_transform, download=True)
cifar_7 = [img for img, label in cifar_dataset if label == 0]
dataloader_cifar = DataLoader(cifar_7, batch_size=BATCH_SIZE, shuffle=True)
unet = UNet(depth=4, time_emb_dim=32, color_channels=3)
unet = to_device(unet)

optimizer = torch.optim.Adam(unet.parameters(), lr=LR)

t_total = time.time()
t = time.time()
with Progress(SpinnerColumn(), *Progress.get_default_columns(), "[yellow]{task.fields[loss]}") as progress:

    epoch_task = progress.add_task("[red]Training...", total=EPOCHS, loss="")
    best_loss = math.inf

    for epoch in range(EPOCHS):

        batch_task = progress.add_task(f"Epoch {epoch}", total=len(dataloader_cifar), loss="")

        for batch_idx, batch in enumerate(dataloader_cifar):

            progress.update(batch_task, advance=1)
            progress.update(epoch_task, advance=1/len(dataloader_cifar))

            if len(batch) != BATCH_SIZE:
                continue

            ## apply random horizontal flip
            if torch.rand(1) > 0.5:
                batch = torch.flip(batch, [3])

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

            if batch_idx % 10 == 0:
                progress.update(batch_task, loss=f"Loss: {loss.item():.4f}")

            if (epoch+1) % 50 == 0 and batch_idx == 0:
                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(denormalize_img(batch[0].permute(1, 2, 0).detach().cpu().squeeze()))
                ax[1].imshow(denormalize_img(noises[0].permute(1, 2, 0).detach().cpu().squeeze()))
                ax[2].imshow(denormalize_img(output[0].permute(1, 2, 0).detach().cpu().squeeze()))
                fig.suptitle(f"Epoch {epoch+1}")
                plt.show()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(unet.state_dict(), f"models/tmp_best.pth")

        progress.update(batch_task, visible=False)


print("Total duration", time.time() - t_total)
torch.save(unet.state_dict(), f"models/unet_{T_MAX}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pth")
test_DDPM_chain(unet, BETA, T_MAX, shape=(1, 3, 32, 32))
