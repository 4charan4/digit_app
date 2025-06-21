import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# --- Define the full CVAE model (same as in training) ---
class CVAE(nn.Module):
    def __init__(self, latent_dim=20, num_classes=10):
        super(CVAE, self).__init__()
        self.label_emb = nn.Embedding(num_classes, 10)
        self.fc1 = nn.Linear(784 + 10, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim + 10, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x, y):
        y = self.label_emb(y)
        x = torch.cat([x.view(-1, 784), y], dim=1)
        h = torch.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y = self.label_emb(y)
        z = torch.cat([z, y], dim=1)
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# --- Load the trained model ---
model = CVAE()
model.load_state_dict(torch.load("cvae_mnist.pth", map_location="cpu"))
model.eval()

# --- Streamlit UI ---
st.title("🧠 MNIST Handwritten Digit Generator")
st.markdown("Generate 5 different handwritten images for a digit using a Conditional VAE model trained on MNIST.")

digit = st.selectbox("Choose a digit (0–9):", list(range(10)))

if st.button("Generate 5 Images"):
    st.subheader(f"Generated Images for Digit: {digit}")
    cols = st.columns(5)
    for i in range(5):
        z = torch.randn(1, 20)
        label = torch.tensor([digit])
        with torch.no_grad():
            img = model.decode(z, label).view(28, 28).numpy()
        cols[i].image(img, width=100, clamp=True)
