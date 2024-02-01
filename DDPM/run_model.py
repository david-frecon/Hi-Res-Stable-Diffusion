import torch
import matplotlib.pyplot as plt

from utils import test_chain

MODEL_DATE = "2024-02-01_22:50:52"
T_MAX = 400
BETA = 0.0001

model = torch.load(f"models/unet_{T_MAX}_{MODEL_DATE}.pth")
model.eval()
test_chain(model, BETA, T_MAX, shape=(1, 3, 32, 32), n_samples=4)
