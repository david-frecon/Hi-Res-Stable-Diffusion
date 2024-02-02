import torch
import matplotlib.pyplot as plt

from utils import to_device, get_device_name
from VAE import VAE
import numpy as np


net = to_device(VAE())
net.load_state_dict(torch.load("models/vae_louis.pt", map_location=get_device_name()))


@torch.no_grad()
def plot_latent_space(vae, n=30, figsize=15):
    digit_size = 512
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n, 3), dtype=np.uint8)

    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = to_device(torch.randn((1, 16**2)))
            x_decoded = vae.dec(z_sample.float()).cpu()
            x_decoded = (x_decoded + 1) / 2
            digit = x_decoded[0].reshape(3, digit_size, digit_size)
            digit = digit.permute(1, 2, 0).numpy()
            digit = digit * 255
            digit = digit.astype(np.uint8)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure)
    plt.show()


plot_latent_space(net)
