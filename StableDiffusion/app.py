import tkinter as tk
from tkinter import ttk
import os
import sys

# module_path = os.path.abspath(os.getcwd())
# if module_path not in sys.path:
#     sys.path.append(module_path)

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import torch

from utils import test_stable_diffusion_chain, test_DDPM_chain, to_device, get_device_name
from UNet.UNetText import UNetText
from UNet.UNet import UNet
from VAE.VAE import VAE
from fashion_clip.fashion_clip import FashionCLIP

BATCH_SIZE = 64
BETA = 0.0001
T_MAX = 1000

STABLE_DIFFUSION_MODELS = {
    "unet": "ldm.pth",
    "vae": "small_vae.pt"
}
DDPM_MODELS = {
    "unet": "ddpm_horses.pth"
}
VAE_MODELS = {
    "vae": "small_vae.pt"
}

def create_app():
    window = tk.Tk()
    window.title("EPITA Stable Diffusion")
    window.geometry("1000x800")

    def update_ui(*args):
        if mode.get() == "DDPM":
            description_label.config(text="Générez des chevaux", font=("Arial", 14))
            description_entry.pack_forget()
        else:
            description_label.config(text="Description:", font=("Arial", 14))
            description_entry.pack(after=description_label)

    def ui_callback(progress, new_image: Figure = None):
        progress_bar["value"] = progress
        if new_image is not None:
            canvas.draw()
        window.update()

    def generate():
        number_of_images = int(number_selector.get())
        description = description_entry.get()
        model = mode.get()
        progress_bar["value"] = 0

        if model == "StableDiffusion":
            f_clip = FashionCLIP("fashion-clip")
            texts_embeddings = to_device(torch.tensor(f_clip.encode_text([description] * number_of_images, batch_size=32))).view(number_of_images, 512)

            unet = UNetText(depth=4, time_emb_dim=32, text_emb_dim=512, color_channels=1)
            unet.load_state_dict(torch.load(f"../models/{STABLE_DIFFUSION_MODELS['unet']}", map_location=get_device_name()))
            unet = to_device(unet)
            unet.eval()
            unet.requires_grad_(False)

            vae = VAE(16**2)
            vae.load_state_dict(torch.load(f"../models/{STABLE_DIFFUSION_MODELS['vae']}", map_location=get_device_name()))
            vae = to_device(vae)
            vae.eval()
            vae.requires_grad_(False)

            test_stable_diffusion_chain(unet, vae, BETA, T_MAX, texts_embeddings, 16, save_video=False, callback=ui_callback, fig=fig)

        elif model == "DDPM":
            unet = UNet(depth=4, time_emb_dim=32, color_channels=3)
            unet.load_state_dict(torch.load(f"../models/{DDPM_MODELS['unet']}", map_location=get_device_name()))
            unet = to_device(unet)
            unet.eval()
            unet.requires_grad_(False)

            test_DDPM_chain(unet, BETA, T_MAX, (number_of_images, 3, 32, 32), number_of_images, save_video=False, callback=ui_callback, fig=fig)

    left_panel = tk.Frame(window)
    left_panel.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

    mode_label = tk.Label(left_panel, text="Modèle", font=("Arial", 14))
    mode_label.pack(anchor=tk.W)

    mode = tk.StringVar(value="StableDiffusion")
    mode.trace_add("write", update_ui)
    modes = ["StableDiffusion", "DDPM"]
    for m in modes:
        tk.Radiobutton(left_panel, text=m, variable=mode, value=m).pack(anchor=tk.W)

    spacer = tk.Label(left_panel, text="")
    spacer.pack()

    number_label = tk.Label(left_panel, text="Nombre d'images:", font=("Arial", 14))
    number_label.pack(anchor=tk.W)
    number_selector = ttk.Spinbox(left_panel, from_=4, to=16, width=5)
    number_selector.set(4)
    number_selector.pack(anchor=tk.W)

    center_frame = tk.Frame(window)
    center_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=20)

    fig = Figure(figsize=(5, 4), dpi=100)
    plot = fig.add_subplot(111)
    plot.axis("off")
    plot.plot()

    canvas = FigureCanvasTkAgg(fig, master=center_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=15)

    description_label = tk.Label(center_frame, text="Description", font=("Arial", 14))
    description_label.pack()
    description_entry = tk.Entry(center_frame, width=50)
    description_entry.pack()

    generate_button = tk.Button(center_frame, text="Générer", command=generate)
    generate_button.pack(pady=10)

    progress_bar = ttk.Progressbar(center_frame, orient="horizontal", mode="determinate", length=300)
    progress_bar.pack(pady=10)

    window.mainloop()


if __name__ == "__main__":
    create_app()
