from functools import cache

import torch
from matplotlib import pyplot as plt

from DDPM.NoiseScheduler import denormalize_img


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
def test_chain(model, max_t, shape=(1, 1, 28, 28), n_samples=4):
    big_chain = []
    for i in range(n_samples):
        full_noise = torch.randn(shape)
        full_noise = to_device(full_noise)
        chain = [full_noise]

        for t in range(max_t):
            predicted_noise = model(chain[-1], torch.tensor([t]).to(get_device()).float())
            new_img = chain[-1] - predicted_noise
            chain.append(new_img)
        big_chain.append(chain)

    fig, ax = plt.subplots(n_samples, 5)
    for i in range(n_samples):
        for t in range(4):
            ax[i, t].imshow(denormalize_img(big_chain[i][t].detach().cpu().numpy().squeeze()), cmap='gray')
        ax[i, 4].imshow(denormalize_img(big_chain[i][-1].detach().cpu().numpy().squeeze()), cmap='gray')

    plt.show()
    return big_chain
