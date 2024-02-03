import os

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd

from NoiseScheduler import betas_schedule, alphas_schedule, alphas_bar_schedule, noise_batch
from utils import to_device, get_device_name
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


vae = VAE()
vae.load_state_dict(torch.load("models/vae_louis.pt", map_location=get_device_name()))
vae = to_device(vae)
vae.eval()
vae.requires_grad_(False)


# Create tensor of all images
images_tensor = torch.zeros(len(df), 1, 16, 16)
for i in range(len(df)):
    img_name = os.path.join(img_dir, 'classes', str(df.iloc[i, 0]) + '.jpg')
    image = Image.open(img_name)
    image = to_device(transforms.ToTensor()(image)).view(1, 3, 512, 512)
    image, _, _ = vae.enc(image)
    images_tensor[i] = image.view(1, 1, 16, 16)

# Create tensor of all descriptions
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


# Transformations pour les images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)
])

# Création du dataset personnalisé
dataset = CustomImageDataset()

# DataLoader
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

unet = UNetText(depth=4, time_emb_dim=32, text_emb_dim=512, color_channels=1)
unet = to_device(unet)

optimizer = torch.optim.Adam(unet.parameters(), lr=LR)

ALPHAS_BAR = to_device(ALPHAS_BAR)

for epoch in range(EPOCHS):

    for batch_idx, (batch_images, batch_descriptions) in enumerate(loader):

        if len(batch_images) != BATCH_SIZE:
            continue

        batch_descriptions = to_device(batch_descriptions)
        latent_batch = batch_images

        # if torch.rand(1) > 0.5:
        #     batch_images = torch.flip(batch_images, [3])

        tensor_t = torch.randint(0, T_MAX, (BATCH_SIZE,))
        tensor_t_float = to_device(tensor_t.float())

        latent_noised_batch, latent_noises = noise_batch(latent_batch, ALPHAS_BAR, tensor_t)

        optimizer.zero_grad()

        output = unet(latent_noised_batch, tensor_t_float, batch_descriptions)
        loss = F.mse_loss(output, latent_noises)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, batch {batch_idx}, loss {loss.item()}")

    print(f"Epoch {epoch}, loss {loss.item()}")
torch.save(unet.state_dict(), f"models/unet_text_{T_MAX}.pt")