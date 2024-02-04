from functools import cache

import torch
from matplotlib import pyplot as plt

from NoiseScheduler.NoiseScheduler import denormalize_img, alphas_schedule, betas_schedule, alphas_bar_schedule


@cache
def get_device_name():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


@cache
def get_device():
    return torch.device(get_device_name())


def to_device(model):
    return model.to(get_device())


@torch.no_grad()
def test_DDPM_chain(model, beta, max_t, shape=(1, 1, 28, 28), n_samples=4):
    big_chain = []

    betas = betas_schedule(beta, max_t)
    alphas = alphas_schedule(betas)
    alphas_bar = alphas_bar_schedule(alphas)

    inverse_sqrt_alphas = 1 / torch.sqrt(alphas)
    inverse_sqrt_alphas_bar = (1 - alphas) / torch.sqrt(1 - alphas_bar)

    for i in range(n_samples):
        full_noise = torch.randn(shape)
        full_noise = to_device(full_noise)
        chain = [full_noise]

        for t in range(max_t - 1, -1, -1):
            predicted_noise = model(chain[-1], torch.tensor([t]).to(get_device()).float())
            new_img = inverse_sqrt_alphas[t] * (chain[-1] - inverse_sqrt_alphas_bar[t] * predicted_noise)
            if t > 0:
                noise = torch.randn(shape)
                noise = to_device(noise)
                new_img = new_img + torch.sqrt(betas[t]) * noise
            new_img = torch.clip(new_img, -1, 1)
            chain.append(new_img)
        big_chain.append(chain)

    fig, ax = plt.subplots(n_samples, 5)
    for i in range(n_samples):
        ax[i, 0].imshow(denormalize_img(big_chain[i][-16].permute(0, 2, 3, 1).detach().cpu().squeeze()))
        ax[i, 1].imshow(denormalize_img(big_chain[i][-8].permute(0, 2, 3, 1).detach().cpu().squeeze()))
        ax[i, 2].imshow(denormalize_img(big_chain[i][-4].permute(0, 2, 3, 1).detach().cpu().squeeze()))
        ax[i, 3].imshow(denormalize_img(big_chain[i][-2].permute(0, 2, 3, 1).detach().cpu().squeeze()))

        ax[i, 4].imshow(denormalize_img(big_chain[i][-1].permute(0, 2, 3, 1).detach().cpu().squeeze()))

    plt.show()
    return big_chain


def test_stable_diffusion_chain(unet, vae, beta, max_t, texts_embeddings, latent_width=16):
    n_samples = len(texts_embeddings)

    latent_shape = (n_samples, 1, latent_width, latent_width)

    betas = betas_schedule(beta, max_t)
    alphas = alphas_schedule(betas)
    alphas_bar = alphas_bar_schedule(alphas)

    inverse_sqrt_alphas = 1 / torch.sqrt(alphas)
    inverse_sqrt_alphas_bar = (1 - alphas) / torch.sqrt(1 - alphas_bar)

    latent_noised = torch.randn(latent_shape)
    latent_noised = to_device(latent_noised)

    chain = [latent_noised]

    for t in range(max_t - 1, -1, -1):

        if t % 100 == 0:
            print("t =", t)

        time_tensor = torch.tensor([t]).to(get_device()).float()
        time_tensor = time_tensor.repeat(n_samples, 1)

        predicted_noise = unet(chain[-1], time_tensor, texts_embeddings)
        new_images = inverse_sqrt_alphas[t] * (chain[-1] - inverse_sqrt_alphas_bar[t] * predicted_noise)

        if t > 0:
            latent_noise = torch.randn(latent_shape)
            latent_noise = to_device(latent_noise)
            new_images = new_images + torch.sqrt(betas[t]) * latent_noise

        new_images = torch.clip(new_images, -1, 1)
        chain.append(new_images)

    fig, ax = plt.subplots(n_samples, 5)
    for i in range(n_samples):
        images = [chain[0][i], chain[500][i], chain[250][i], chain[100][i], chain[-1][i]]
        decoded_images = [vae.dec(img.view(1, latent_width**2)).cpu().detach().squeeze() for img in images]
        decoded_images = [denormalize_img(img.permute(1, 2, 0)) for img in decoded_images]
        for j, img in enumerate(decoded_images):
            ax[i, j].imshow(img)
            ax[i, j].axis("off")

    plt.show()
    return chain
