
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import Generator, Discriminator

# Hyperparameters
z_dim = 100
batch_size = 128
lr = 2e-4
epochs = 20
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Models
G = Generator(z_dim).to(device)
D = Discriminator().to(device)

# Loss & Optimizers
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

for epoch in range(epochs):
    for real, _ in loader:
        real = real.view(real.size(0), -1).to(device)
        batch_size = real.size(0)

        # ======================
        # Train Discriminator
        # ======================
        z = torch.randn(batch_size, z_dim).to(device)
        fake = G(z)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        loss_D = (
            criterion(D(real), real_labels) +
            criterion(D(fake.detach()), fake_labels)
        )

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ======================
        # Train Generator
        # ======================
        loss_G = criterion(D(fake), real_labels)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")
