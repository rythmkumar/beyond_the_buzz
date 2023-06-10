import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

batch_size=100
epochs=100
learning_rate = 0.0002
image_size=784
latent_size=64
hidden_size=256

class Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.model=nn.Sequential(
    nn.Linear(latent_size,hidden_size ),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())
  
  def forward(self,x):
    img=self.model(x)
    img=img.view(-1,784)
    return img
    
class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.model=nn.Sequential(nn.Linear(784,256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid()
  )
  def forward(self, img):
    img_flat = img.view(img.size(0), -1)
    validity = self.model(img_flat)
    return validity

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset=mnist_data, batch_size=batch_size, shuffle=True)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


G=Generator()
D=Discriminator()
loss=nn.BCELoss()
d_optimizer = optim.Adam(D.model.parameters(), lr=learning_rate)
g_optimizer = optim.Adam(G.model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for batch_idx, (real_images, _) in enumerate(tqdm(data_loader)):
        real_images = real_images.view(-1, image_size)
        batch_size = real_images.size(0)
        
        # Labels for real and fake images
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # Train Discriminator
        
        d_optimizer.zero_grad()
        outputs = D(real_images)
        d_loss_real = loss(outputs, real_labels)
        real_score = outputs
        
        z = torch.randn(batch_size, latent_size)
        fake_images = G(z)
        outputs = D(fake_images.detach())
        d_loss_fake = loss(outputs, fake_labels)
        fake_score = outputs
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        z = torch.randn(batch_size, latent_size)
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = loss(outputs, real_labels)
        g_loss.backward()
        g_optimizer.step()
        
    # Print loss and save generated images
    print(f"Epoch [{epoch+1}/{epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")
    
import os
from IPython.display import Image
from torchvision.utils import save_image
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
    
sample_vectors = torch.randn(batch_size, latent_size)

def save_fake_images(index):
    fake_images = G(sample_vectors)
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    print('Saving', fake_fname)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=10)
    
# After training
save_fake_images(0)
Image(os.path.join(sample_dir, 'fake_images-0000.png'))
