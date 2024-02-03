import os

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd

from NoiseScheduler import betas_schedule, alphas_schedule, alphas_bar_schedule, noise_batch
from utils import to_device
from UNetText import UNetText
from VAE import VAE

from fashion_clip.fashion_clip import FashionCLIP

f_clip = FashionCLIP("fashion-clip")

BATCH_SIZE = 64
EPOCHS = 10
LR = 0.0001
BETA = 0.0001
T_MAX = 1000
BETAS = betas_schedule(BETA, T_MAX)
ALPHAS = alphas_schedule(BETAS)
ALPHAS_BAR = alphas_bar_schedule(ALPHAS)

img_dir = 'data/data_for_fashion_clip/out/'
csv_file = 'data/data_for_fashion_clip/articles.csv'

df = pd.read_csv(csv_file)
df['detail_desc'] = df['detail_desc'].astype(str)
texts = df['detail_desc'].to_list()
texts_embeddings = f_clip.encode_text(texts, batch_size=32)

embeddings_dict = {}
for i, text in enumerate(texts):
    embeddings_dict[text] = texts_embeddings[i]


class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.csv_data = pd.read_csv(csv_file)
        self.csv_data['detail_desc'] = self.csv_data['detail_desc'].astype(str)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, 'classes', str(self.csv_data.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name)
        description = self.csv_data.iloc[idx, 24]
        description = embeddings_dict[description]

        if self.transform:
            image = self.transform(image)

        return image, description

# Transformations pour les images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)
])

# Création du dataset personnalisé
dataset = CustomImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)

# DataLoader
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

unet = UNetText(depth=4, time_emb_dim=32, text_emb_dim=512, color_channels=1)
unet = to_device(unet)

vae = VAE()
vae.load_state_dict(torch.load("models/vae_louis.pt", map_location=torch.device('mps')))
vae = to_device(vae)
vae.eval()
vae.requires_grad_(False)

optimizer = torch.optim.Adam(unet.parameters(), lr=LR)

for epoch in range(EPOCHS):

    for batch_idx, (batch_images, batch_descriptions) in enumerate(loader):

        if len(batch_images) != BATCH_SIZE:
            continue

        #batch_images = batch_images
        batch_descriptions = to_device(batch_descriptions)

        if torch.rand(1) > 0.5:
            batch_images = torch.flip(batch_images, [3])

        tensor_t = torch.randint(0, T_MAX, (BATCH_SIZE,))
        tensor_t_float = to_device(tensor_t.float())
        noised_batch, noises = noise_batch(batch_images, ALPHAS_BAR, tensor_t)
        noises = to_device(noises)
        noised_batch = to_device(noised_batch)

        latent_noised_batch, _, _ = vae.enc(noised_batch)
        latent_noised_batch = to_device(latent_noised_batch)
        latent_noises, _, _ = vae.enc(noises)
        latent_noises = to_device(latent_noises)

        optimizer.zero_grad()

        latent_noised_batch = latent_noised_batch.view(BATCH_SIZE, 1, 16, 16)
        latent_noises = latent_noises.view(BATCH_SIZE, 1, 16, 16)
        output = unet(latent_noised_batch, tensor_t_float, batch_descriptions)
        loss = F.mse_loss(output, latent_noises)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, batch {batch_idx}, loss {loss.item()}")

    print(f"Epoch {epoch}, loss {loss.item()}")
torch.save(unet.state_dict(), f"models/unet_text_{T_MAX}.pt")