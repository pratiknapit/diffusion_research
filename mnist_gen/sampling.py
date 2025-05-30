from vae_model import VAE
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE().to(device)
vae.load_state_dict(torch.load("vae_mnist_1.pt", map_location=device))
vae.eval()

z = torch.rand(vae.latent_dim)
sample_x = vae.decode(z)
sample_x = sample_x.squeeze().view(28, 28)

plt.imshow(sample_x.detach().cpu().numpy(), cmap="gray")
plt.title("Grayscale Image of a 1")
plt.axis("off")
plt.show()