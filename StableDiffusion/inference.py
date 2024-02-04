import math

import torch

from UNet.UNetText import UNetText
from utils import to_device, get_device_name, test_stable_diffusion_chain
from VAE.VAE import VAE
from fashion_clip.fashion_clip import FashionCLIP

UNET_MODEL_NAME = "unet_text_color.pth"
VAE_MODEL_NAME = "vae_louis.pt"
VAE_LATENT_DIM = 16**2
VAE_LATENT_WIDTH = int(math.sqrt(VAE_LATENT_DIM))

BATCH_SIZE = 64
BETA = 0.0001
T_MAX = 1000

MY_TEXTS = ["a red dress", "blue T-Shirt", "pink tshirt", "a yellow jean", "black dress"]
#MY_TEXTS = ["a red dress", "blue T-Shirt"]

f_clip = FashionCLIP("fashion-clip")
texts_embeddings = to_device(torch.tensor(f_clip.encode_text(MY_TEXTS, batch_size=32))).view(len(MY_TEXTS), 512)

unet = UNetText(depth=4, time_emb_dim=32, text_emb_dim=512, color_channels=1)
unet.load_state_dict(torch.load(f"../models/{UNET_MODEL_NAME}", map_location=get_device_name()))
unet = to_device(unet)
unet.eval()
unet.requires_grad_(False)

vae = VAE(latent_dim=VAE_LATENT_DIM)
vae.load_state_dict(torch.load(f"../models/{VAE_MODEL_NAME}", map_location=get_device_name()))
vae = to_device(vae)
vae.eval()
vae.requires_grad_(False)


input_shape = (1, 3, 512, 512)
latent_shape = (1, 1, VAE_LATENT_WIDTH, VAE_LATENT_WIDTH)

test_stable_diffusion_chain(unet, vae, BETA, T_MAX, texts_embeddings, VAE_LATENT_WIDTH, save_video=True)
