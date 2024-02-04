import math
import os

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd

from rich.progress import SpinnerColumn
from rich.progress import Progress

from fashion_clip.fashion_clip import FashionCLIP

from NoiseScheduler.NoiseScheduler import betas_schedule, alphas_schedule, alphas_bar_schedule, noise_batch
from utils import to_device, get_device_name, test_stable_diffusion_chain
from UNet.UNetText import UNetText
from VAE.VAE import VAE

MODEL_NAME = "unet_text_color.pt"
VAE_MODEL_NAME = "vae_louis.pt"
VAE_LATENT_DIM = 16**2
IMG_DIR = '../data/data_for_fashion_clip/out/'
CSV_FILE = '../data/data_for_fashion_clip/articles.csv'

BATCH_SIZE = 128
EPOCHS = 100
LR = 0.0001
BETA = 0.0001
T_MAX = 1000

BETAS = betas_schedule(BETA, T_MAX)
ALPHAS = alphas_schedule(BETAS)
ALPHAS_BAR = to_device(alphas_bar_schedule(ALPHAS))

vae_latent_width = int(math.sqrt(VAE_LATENT_DIM))

f_clip = FashionCLIP("fashion-clip")
df = pd.read_csv(CSV_FILE)

df['detail_desc'] = df['detail_desc'].astype(str)
df['colour_group_name'] = df['colour_group_name'].astype(str)
texts = df['detail_desc'].to_list()
colors = df['colour_group_name'].to_list()
full_texts = [f"{color} {text}" for text, color in zip(texts, colors)]

texts_embeddings = f_clip.encode_text(full_texts, batch_size=32)
test_texts = ["blue T-Shirt", "Jean", "Yellow Jacket"]
test_texts_embeddings = f_clip.encode_text(test_texts, batch_size=32)


vae = VAE(latent_dim=VAE_LATENT_DIM)
vae.load_state_dict(torch.load(f"../models/{VAE_MODEL_NAME}", map_location=get_device_name()))
vae = to_device(vae)
vae.eval()
vae.requires_grad_(False)


images_tensor = torch.zeros(len(df), 1, vae_latent_width, vae_latent_width)
for i in range(len(df)):
    img_name = os.path.join(IMG_DIR, 'classes', str(df.iloc[i, 0]) + '.jpg')
    image = Image.open(img_name)
    image = to_device(transforms.ToTensor()(image)).view(1, 3, 512, 512)
    image, _, _ = vae.enc(image)
    images_tensor[i] = image.view(1, 1, vae_latent_width, vae_latent_width)


descriptions_tensor = torch.zeros(len(df), 512)
for i in range(len(df)):
    descriptions_tensor[i] = torch.tensor(texts_embeddings[i])


images_tensor = to_device(images_tensor)
descriptions_tensor = to_device(descriptions_tensor)


class CustomImageDataset(Dataset):
    def __len__(self):
        return len(images_tensor)

    def __getitem__(self, idx):
        return images_tensor[idx], descriptions_tensor[idx]


dataset = CustomImageDataset()
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

unet = UNetText(depth=4, time_emb_dim=32, text_emb_dim=512, color_channels=1)
unet = to_device(unet)

optimizer = torch.optim.Adam(unet.parameters(), lr=LR)

with Progress(SpinnerColumn(), *Progress.get_default_columns(), "[yellow]{task.fields[loss]}") as progress:

    epoch_task = progress.add_task("[red]Training...", total=EPOCHS, loss="")
    best_loss = math.inf

    for epoch in range(EPOCHS):

        batch_task = progress.add_task(f"Epoch {epoch}", total=len(loader), loss="")

        for batch_idx, (batch_images, batch_descriptions) in enumerate(loader):

            progress.update(batch_task, advance=1)
            progress.update(epoch_task, advance=1 / len(loader))

            if len(batch_images) != BATCH_SIZE:
                continue

            batch_descriptions = to_device(batch_descriptions)
            latent_batch = batch_images

            tensor_t = torch.randint(0, T_MAX, (BATCH_SIZE,))
            tensor_t_float = to_device(tensor_t.float())

            latent_noised_batch, latent_noises = noise_batch(latent_batch, ALPHAS_BAR, tensor_t)

            optimizer.zero_grad()

            output = unet(latent_noised_batch, tensor_t_float, batch_descriptions)
            loss = F.mse_loss(output, latent_noises)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                progress.update(batch_task, loss=f"Loss: {loss.item():.4f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(unet.state_dict(), f"../models/tmp_best.pth")

        progress.update(batch_task, visible=False)

torch.save(unet.state_dict(), f"../models/{MODEL_NAME}")
test_stable_diffusion_chain(unet, vae, BETA, T_MAX, to_device(torch.tensor(test_texts_embeddings)), latent_width=vae_latent_width)
