import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Define decoder part of the VAE
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

# Load model
model = VAE()
model.load_state_dict(torch.load("vae_mnist.pth", map_location='cpu'))
model.eval()

# UI
st.title("MNIST Handwritten Digit Generator")
st.write("Generate handwritten digits using a trained VAE model.")

digit = st.selectbox("Choose a digit (0–9):", list(range(10)))

if st.button("Generate 5 Images"):
    st.subheader(f"Generated Images for digit: {digit}")
    cols = st.columns(5)
    for i in range(5):
        z = torch.randn(1, 20)  # sample latent vector
        with torch.no_grad():
            img = model.decode(z).view(28, 28).numpy()
        cols[i].image(img, width=100, clamp=True)
 
