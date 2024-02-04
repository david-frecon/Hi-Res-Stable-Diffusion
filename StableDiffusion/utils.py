from functools import cache

import numpy as np
from rich.progress import SpinnerColumn
from rich.progress import Progress

import torch
from matplotlib import pyplot as plt

from StableDiffusion.NoiseScheduler.NoiseScheduler import denormalize_img, alphas_schedule, betas_schedule, alphas_bar_schedule


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


def get_grid_size(n_samples):
    if n_samples == 4:
        return 2, 2
    if n_samples == 5 or n_samples == 6:
        return 2, 3
    if n_samples == 7 or n_samples == 8:
        return 2, 4
    if n_samples == 9:
        return 3, 3
    if n_samples == 10:
        return 2, 5
    if n_samples == 11 or n_samples == 12:
        return 3, 4
    if n_samples == 13 or n_samples == 14:
        return 3, 5
    if n_samples == 15 or n_samples == 16:
        return 4, 4
    return 1, 1


@torch.no_grad()
def test_DDPM_chain(model, beta, max_t, shape=(1, 1, 28, 28), n_samples=4, save_video=False, callback=None, fig=None):
    betas = betas_schedule(beta, max_t)
    alphas = alphas_schedule(betas)
    alphas_bar = alphas_bar_schedule(alphas)

    inverse_sqrt_alphas = 1 / torch.sqrt(alphas)
    inverse_sqrt_alphas_bar = (1 - alphas) / torch.sqrt(1 - alphas_bar)

    full_noise = torch.randn(shape)
    full_noise = to_device(full_noise)
    chain = [full_noise]

    for t in range(max_t - 1, -1, -1):
        predicted_noise = model(chain[-1], torch.tensor([t] * n_samples).to(get_device()).float())
        new_img = inverse_sqrt_alphas[t] * (chain[-1] - inverse_sqrt_alphas_bar[t] * predicted_noise)
        if t > 0:
            noise = torch.randn(shape)
            noise = to_device(noise)
            new_img = new_img + torch.sqrt(betas[t]) * noise
        new_img = torch.clip(new_img, -1, 1)
        chain.append(new_img)

        if t % 10 == 0 and callback is not None:
            l, c = get_grid_size(n_samples)
            fig.clf()
            ax = fig.subplots(l, c)
            images = [chain[-1][i] for i in range(n_samples)]
            decoded_images = [denormalize_img(img.permute(1, 2, 0).detach().cpu().squeeze()) for img in images]
            for j, img in enumerate(decoded_images):
                ax_index = (j // c, j % c)
                ax[ax_index].imshow(img)
                ax[ax_index].axis("off")
            callback(100 - t / max_t * 100, fig)

    if callback is None:
        fig, ax = plt.subplots(n_samples, 5)
        for i in range(n_samples):
            for j, t in enumerate([-16, -8, -4, -2, -1]):
                ax[i, j].imshow(denormalize_img(chain[t][i].permute(0, 2, 3, 1).detach().cpu().squeeze()))
                ax[i, j].axis("off")

        plt.show()

    if save_video:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (512 * n_samples, 512))
        img = np.zeros((512, 512 * n_samples, 3), dtype=np.uint8)
        for t in range(max_t // 2, max_t):
            images = [chain[t][i] for i in range(n_samples)]
            decoded_images = [denormalize_img(img.permute(0, 2, 3, 1).detach().cpu().squeeze()).astype(np.uint8) for img in images]
            decoded_images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in decoded_images]
            decoded_images = [cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST) for img in decoded_images]

            for j, decoded_img in enumerate(decoded_images):
                img[:, 512 * j:512 * (j + 1), :] = decoded_img

            out.write(img)
        out.release()

    return chain


def test_stable_diffusion_chain(unet, vae, beta, max_t, texts_embeddings, latent_width=16, save_video=False, callback=None, fig=None):
    n_samples = len(texts_embeddings)

    latent_shape = (n_samples, 1, latent_width, latent_width)

    betas = betas_schedule(beta, max_t)
    alphas = alphas_schedule(betas)
    alphas_bar = alphas_bar_schedule(alphas)

    inverse_sqrt_alphas = 1 / torch.sqrt(alphas)
    inverse_sqrt_alphas_bar = (1 - alphas) / torch.sqrt(1 - alphas_bar)

    with Progress(SpinnerColumn(), *Progress.get_default_columns()) as progress:

        latent_noised = torch.randn(latent_shape)
        latent_noised = to_device(latent_noised)

        chain = [latent_noised]

        time_task = progress.add_task("[red]Sampling...", total=max_t)

        for t in range(max_t - 1, -1, -1):
            progress.update(time_task, advance=1)

            time_tensor = torch.tensor([t for _ in range(n_samples)]).to(get_device()).float()
            predicted_noise = unet(chain[-1], time_tensor, texts_embeddings)
            new_img = inverse_sqrt_alphas[t] * (chain[-1] - inverse_sqrt_alphas_bar[t] * predicted_noise)

            if t > 0:
                latent_noise = torch.randn(latent_shape)
                latent_noise = to_device(latent_noise)
                new_img = new_img + torch.sqrt(betas[t]) * latent_noise

            new_img = torch.clip(new_img, -1, 1)
            chain.append(new_img)

            if t % 10 == 0 and callback is not None:
                l, c = get_grid_size(n_samples)
                fig.clf()
                ax = fig.subplots(l, c)
                images = [chain[-1][i] for i in range(n_samples)]
                decoded_images = [vae.dec(img.view(1, 16 * 16)).cpu().detach().squeeze() for img in images]
                decoded_images = [denormalize_img(img.permute(1, 2, 0)).astype(int) for img in decoded_images]
                for j, img in enumerate(decoded_images):
                    ax_index = (j // c, j % c)
                    ax[ax_index].imshow(img)
                    ax[ax_index].axis("off")
                callback(100 - t / max_t * 100, fig)

        progress.update(time_task, visible=False)

        if callback is None:
            fig, ax = plt.subplots(n_samples, 5)
            for i in range(n_samples):
                images = [chain[0][i], chain[-500][i], chain[-250][i], chain[-100][i], chain[-1][i]]
                decoded_images = [vae.dec(img.view(1, 16 * 16)).cpu().detach().squeeze() for img in images]
                decoded_images = [denormalize_img(img.permute(1, 2, 0)) for img in decoded_images]
                for j, img in enumerate(decoded_images):
                    ax[i, j].imshow(img)
                    ax[i, j].axis("off")
            plt.show()

    if save_video:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (512 * n_samples, 512))
        img = np.zeros((512, 512 * n_samples, 3), dtype=np.uint8)
        for t in range(max_t // 2, max_t):
            images = [chain[t][i] for i in range(n_samples)]
            decoded_images = [vae.dec(img.view(1, 16 * 16)).cpu().detach().squeeze() for img in images]
            decoded_images = [denormalize_img(img.permute(1, 2, 0)).astype(np.uint8) for img in decoded_images]
            decoded_images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in decoded_images]

            for j, decoded_img in enumerate(decoded_images):
                img[:, 512 * j:512 * (j + 1), :] = decoded_img

            out.write(img)
        out.release()

    return chain
