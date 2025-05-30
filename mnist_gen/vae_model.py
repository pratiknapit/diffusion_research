import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        #Encoder
        self.h1 = nn.Linear(input_dim, hidden_dim)
        self.h_mu = nn.Linear(hidden_dim, latent_dim)
        self.h_var = nn.Linear(hidden_dim, latent_dim)

        #Decoder
        self.h2 = nn.Linear(latent_dim, hidden_dim)
        self.h3 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h = F.relu(self.h1(x))
        mu = self.h_mu(h)
        logvar = self.h_var(h)
        return mu, logvar
    
    def reparameterise(self, mu, logvar):
        return mu + torch.exp(0.5*logvar)*torch.rand_like(logvar)
    
    def decode(self, z):
        h = F.relu(self.h2(z))
        return torch.sigmoid(self.h3(h)) # Output a probability between [0, 1]
    

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterise(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

