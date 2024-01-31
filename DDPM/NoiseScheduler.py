import torch
import numpy as np


def normalize_img(img):
    normalize = (img - img.min()) / (img.max() - img.min())
    return normalize * 2 - 1


def denormalize_img(img):
    denormalize = (img + 1) / 2
    return np.clip(denormalize * 255, 0, 255)


def betas_schedule(beta, t_max):
    return torch.linspace(beta, beta * t_max, steps=t_max)


def alphas_schedule(betas):
    return 1 - betas


def alphas_bar_schedule(alphas):
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alpha_bar


def noise_sample(img, alpha_bar_t):
    noise = torch.randn_like(img)
    return torch.sqrt(alpha_bar_t) * img + torch.sqrt(1-alpha_bar_t) * noise
