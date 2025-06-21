import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class CVAE(nn.Module):
    def __init__(self, latent_dim=20, num_classes=10):
        super(CVAE, self).__init__()
        self.label_emb = nn.Embedding(num_classes, 10)
        self.fc3 = nn.Linear(latent_dim + 10, 400)
        self.fc4 = nn.Linear(400, 784)

    def decode(self, z, y):
        y = self.label_emb(y)
        z = torch.cat([z, y], dim=1)
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

# Load model
model = CVAE()
model.load_state_dict(torch.load("cvae_mnist.pth", map_location="cpu"))
model.eval()

st.title("MNIST Digit Generator (CVAE)")
digit = st.selectbox("Choose a digit (0–9):", list(range(10)))

if st.button("Generate 5 Images"):
    st.subheader(f"Generated images for digit {digit}")
    cols = st.columns(5)
    for i in range(5):
        z = torch.randn(1, 20)
        label = torch.tensor([digit])
        with torch.no_grad():
            img = model.decode(z, label).view(28, 28).numpy()
        cols[i].image(img, width=100, clamp=True)
