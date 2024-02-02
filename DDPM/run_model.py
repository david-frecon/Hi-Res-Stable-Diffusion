import torch
import matplotlib.pyplot as plt

from UNet import UNet
from utils import test_chain, to_device

MODEL_DATE = "2024-02-01_22:50:52"
T_MAX = 400
BETA = 0.0001
MODEL_PATH = f"models/unet_{T_MAX}_{MODEL_DATE}.pth"
MODEL_PATH = "models/tmp_best.pth"

#model = torch.load(f"models/unet_{T_MAX}_{MODEL_DATE}.pth")
model = UNet(depth=4, time_emb_dim=32, color_channels=3)
model.load_state_dict(torch.load(MODEL_PATH))
model = to_device(model)

model.eval()
test_chain(model, BETA, T_MAX, shape=(1, 3, 32, 32), n_samples=4)
