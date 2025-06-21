import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Full VAE class from training
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Load full model (to satisfy load_state_dict)
model = VAE()
model.load_state_dict(torch.load("vae_mnist.pth", map_location="cpu"))
model.eval()

# Streamlit UI
st.title("MNIST Digit Generator")
digit = st.selectbox("Choose a digit (0–9):", list(range(10)))

if st.button("Generate 5 Images"):
    st.subheader(f"Generated Images for digit {digit}")
    cols = st.columns(5)
    for i in range(5):
        z = torch.randn(1, 20)  # random latent vector
        with torch.no_grad():
            img = model.decode(z).view(28, 28).numpy()
        cols[i].image(img, width=100, clamp=True)
