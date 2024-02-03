import torch
import matplotlib.pyplot as plt

from NoiseScheduler import betas_schedule, alphas_schedule, alphas_bar_schedule, denormalize_img
from UNetText import UNetText
from utils import to_device, get_device_name, get_device
from VAE import VAE
from fashion_clip.fashion_clip import FashionCLIP

BATCH_SIZE = 64
BETA = 0.0001
T_MAX = 1000

MY_TEXTS = ["a red dress", "blue T-Shirt", "pink tshirt", "a yellow jean", "black dress"]

f_clip = FashionCLIP("fashion-clip")
texts_embeddings = to_device(torch.tensor(f_clip.encode_text(MY_TEXTS, batch_size=32)))
print(texts_embeddings.shape)

unet = UNetText(depth=4, time_emb_dim=32, text_emb_dim=512, color_channels=1)
unet.load_state_dict(torch.load("models/unet_text_1.pth", map_location=get_device_name()))
unet = to_device(unet)
unet.eval()
unet.requires_grad_(False)

vae = VAE()
vae.load_state_dict(torch.load("models/vae_louis.pt", map_location=get_device_name()))
vae = to_device(vae)
vae.eval()
vae.requires_grad_(False)


input_shape = (1, 3, 512, 512)
latent_shape = (1, 1, 16, 16)

big_chain = []
n_samples = 5

betas = betas_schedule(BETA, T_MAX)
alphas = alphas_schedule(betas)
alphas_bar = alphas_bar_schedule(alphas)

inverse_sqrt_alphas = 1 / torch.sqrt(alphas)
inverse_sqrt_alphas_bar = (1 - alphas) / torch.sqrt(1 - alphas_bar)

for i in range(n_samples):
    print("Sample", i + 1, "/", n_samples)
    full_noise = torch.randn(input_shape)
    full_noise = to_device(full_noise)

    latent_noised, _, _ = vae.enc(full_noise)
    latent_noised = to_device(latent_noised).view(latent_shape)

    chain = [latent_noised]

    text_tensor = texts_embeddings[i].view(1, 512)

    for t in range(T_MAX - 1, -1, -1):

        if t % 10 == 0:
            print("t =", t)

        time_tensor = torch.tensor([t]).to(get_device()).float()
        predicted_noise = unet(chain[-1], time_tensor, text_tensor)
        new_img = inverse_sqrt_alphas[t] * (chain[-1] - inverse_sqrt_alphas_bar[t] * predicted_noise)

        if t > 0:
            noise = torch.randn(input_shape)
            noise = to_device(noise)

            latent_noise, _, _ = vae.enc(noise)
            latent_noise = to_device(latent_noise).view(latent_shape)

            new_img = new_img + torch.sqrt(betas[t]) * latent_noise
        new_img = torch.clip(new_img, -1, 1)
        chain.append(new_img)
    big_chain.append(chain)

fig, ax = plt.subplots(n_samples, 5)
for i in range(n_samples):
    images = [big_chain[i][-16], big_chain[i][-8], big_chain[i][-4], big_chain[i][-2], big_chain[i][-1]]
    decoded_images = [vae.dec(img.view(1, 16*16)).cpu().detach().squeeze() for img in images]
    decoded_images = [denormalize_img(img.permute(1, 2, 0)) for img in decoded_images]
    for j, img in enumerate(decoded_images):
        ax[i, j].imshow(img)
        ax[i, j].axis("off")

plt.show()
