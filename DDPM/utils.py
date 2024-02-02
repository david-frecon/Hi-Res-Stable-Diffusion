import math
from functools import cache

import torch
from matplotlib import pyplot as plt

from NoiseScheduler import denormalize_img, alphas_schedule, betas_schedule, alphas_bar_schedule


@cache
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def to_device(model):
    return model.to(get_device())


@torch.no_grad()
def test_chain(model, beta, max_t, shape=(1, 1, 28, 28), n_samples=4):
    big_chain = []

    ##
    betas = betas_schedule(beta, max_t)
    alphas = alphas_schedule(betas)
    alphas_bar = alphas_bar_schedule(alphas)

    ##
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
        # for t in range(4):
        #     ax[i, t].imshow(denormalize_img(big_chain[i][max_t * t // 4].permute(0, 2, 3, 1).detach().cpu().squeeze()))
        ax[i, 4].imshow(denormalize_img(big_chain[i][-16].permute(0, 2, 3, 1).detach().cpu().squeeze()))
        ax[i, 4].imshow(denormalize_img(big_chain[i][-8].permute(0, 2, 3, 1).detach().cpu().squeeze()))
        ax[i, 4].imshow(denormalize_img(big_chain[i][-4].permute(0, 2, 3, 1).detach().cpu().squeeze()))
        ax[i, 4].imshow(denormalize_img(big_chain[i][-2].permute(0, 2, 3, 1).detach().cpu().squeeze()))

        ax[i, 4].imshow(denormalize_img(big_chain[i][-1].permute(0, 2, 3, 1).detach().cpu().squeeze()))

    plt.show()
    return big_chain
