import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib as plt
from vae_model import VAE



# Data
transform = transforms.ToTensor()
mnist_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
print("Data set types: ", type(mnist_dataset[0][0]), type(mnist_dataset[0][1]))

# Input 
number = input("Enter number generator: \n")

# Filter only images where the label is "1"
only_number_indices = [i for i, (_, label) in enumerate(mnist_dataset) if label == int(number)]
print("Only ones data length:", len(only_number_indices))
number_dataset = Subset(mnist_dataset, only_number_indices)
number_loader = DataLoader(number_dataset, batch_size=128, shuffle=True)

# Check data dims
for x, y in number_loader:
    print("Image batch shape:", x.shape)
    print("Label batch shape:", y.shape)
    break  # Just check the first batch


# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# VAE Loss Function
def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(logvar - torch.exp(logvar) - mu.pow(2) + 1)
    return BCE + KLD

# Training
epochs = 100
for epoch in range(epochs):
    vae.train()
    train_loss = 0
    
    for x, _ in number_loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = vae(x)
        loss = vae_loss(recon_x, x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {train_loss / len(number_loader.dataset) :.2f}")

# Save model
torch.save(vae.state_dict(), "vae_mnist_1.pt")