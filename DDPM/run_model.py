import torch
import matplotlib.pyplot as plt

from utils import test_chain

MODEL_DATE = "2024-02-01_20:03:33"
T_MAX = 400

model = torch.load(f"models/unet_{T_MAX}_{MODEL_DATE}.pth")
model.eval()
test_chain(model, T_MAX, n_samples=4)
