import torch
import matplotlib.pyplot as plt
from VAE import VAE
import torchvision.datasets as datasets
import torchvision.transforms as transforms


dataset = datasets.ImageFolder(root='data/data_for_fashion_clip/out/', transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)
]))

net = VAE().cuda()
net.load_state_dict(torch.load("models/vae.pt"))

indexes = [0, 1, 2, 3, 4]
plt.figure(figsize=(10, 8))
for i, index in enumerate(indexes):
    x, _ = dataset[index]
    print(x.max(), x.min())

    with torch.no_grad():
        reconstruction, _, _ = net(x.unsqueeze(0).cuda())
    reconstruction = reconstruction.clip(-1, 1)
    reconstruction = (reconstruction + 1) / 2
    reconstruction = reconstruction * 255
    reconstruction = reconstruction.cpu().numpy().astype(int)
    print(reconstruction.max(), reconstruction.min())

    ax = plt.subplot(len(indexes), 2, i * 2 + 1)
    ax.axis("off")
    ax.set_title("Original")
    xtmp = x.cpu().numpy().transpose(1, 2, 0)
    xtmp = (xtmp + 1) / 2
    ax.imshow(xtmp)
    ax = plt.subplot(len(indexes), 2, i * 2 + 2)
    ax.axis("off")
    ax.set_title("Reconstruction")
    ax.imshow(reconstruction[0].transpose(1, 2, 0))

plt.show()
