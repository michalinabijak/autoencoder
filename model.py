import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset

from typing import List, Optional
from tqdm import tqdm
import itertools

import matplotlib.pyplot as plt
import numpy as np

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load the train and test sets.
train = torchvision.datasets.FashionMNIST(root="/tmp", download=True, transform=torchvision.transforms.ToTensor(),
                                          train=True)
testset = torchvision.datasets.FashionMNIST(root="/tmp", download=True, transform=torchvision.transforms.ToTensor(),
                                            train=False)

# Split train data into training and validation sets.
trainset, validset = torch.utils.data.random_split(train, lengths=[50000, 10000])


def add_noise(x):
    """
    noising function, removes one or two quadrants of an input image, at random
    """
    x = x.detach().clone()
    for _ in (1, 2):
        for i in range(x.size(0)):
            xoffset = np.random.choice([0, 14])
            yoffset = np.random.choice([0, 14])
            x[i, 0, xoffset:xoffset + 14, yoffset:yoffset + 14] = 0
    return x


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        dims = [input_dim] + (hidden_dims or []) + [input_dim]
        self.linear_layers = nn.ModuleList([nn.Linear(d1, d2) for (d1, d2) in zip(dims[:-1], dims[1:])])

    def forward(self, x: torch.Tensor):
        org_size = list(x.size())
        x = x.view(x.size(0), -1)  # flatten the input

        for linear_layer in self.linear_layers[:-1]:
            x = torch.relu(linear_layer(x))
        x = self.linear_layers[-1](x)

        return x.view(*org_size)


def train_autoencoder(input_dim, trainset, validset, n_epochs, batch_size, lr, optimizer,
                      hid_dim: Optional[List[int]] = None):
    """
    Build an AutoEncoder with mean squared loss, given hyperparameters: n_epochs, learning rate - lr, hidden dimensions - hid_dim,
    optimizer in the form troch.optim.name_of_the_optimizer.
    Train on trainset, validate on validset.
    """
    # Build the model
    model = AutoEncoder(input_dim, hid_dim).to(DEVICE)
    loss = torch.nn.MSELoss()

    # Build the optimizer
    optim = optimizer(model.parameters(model), lr)

    # Load datasets
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size)

    # Track loss
    loss_hist = {"train": [], "valid": []}
    best_loss = 1.0

    # Train loop
    for epoch in tqdm(range(n_epochs)):
        train_loss, valid_loss = 0, 0
        for x, _ in trainloader:
            x = x.to(DEVICE)

            optim.zero_grad()
            x_denoise = model(add_noise(x))

            l = loss(x_denoise, x)
            train_loss += l

            l.backward()
            optim.step()

        for x, _ in validloader:
            x = x.to(DEVICE)

            with torch.no_grad():
                x_denoise = model(add_noise(x))
            l = loss(x_denoise, x)
            valid_loss += l.item()

        best_loss = min(best_loss, valid_loss / len(validloader))

        loss_hist["train"].append(train_loss / len(trainloader))
        loss_hist["valid"].append(valid_loss / len(validloader))

    return model, loss_hist, best_loss


# HYPERPARAMETER SEARCH

# Dimension of the input data.
input_size = trainset[0][0].numel()

# Define possible hyperparameter values.

n_epochs = 25
batch_sizes = [8, 16]
hidden_dims = [[784], [784, 784], [784, 256], [784, 784, 784], [784, 512, 512, 784], [784, 784, 512, 784, 784]]
learning_rates = [1e-3, 1e-4, 3e-4]
optimizer = torch.optim.Adam

# Iterate over all combinations of hyperparameters and find the best (returning smallest valid loss) settings.
# Keep track of the best model and its loss history.

best_model, loss_hist, best_params = None, None, None
best_loss = 1
for hid_dim, learning_rate, batch_size in itertools.product(hidden_dims, learning_rates, batch_sizes):
    model, lh, loss = train_autoencoder(input_size, trainset, validset, n_epochs, batch_size, learning_rate, optimizer,
                                        hid_dim)
    print(f"Hidden size: {hid_dim},lr: {learning_rate}, best loss: {loss}")
    if loss < best_loss:
        best_model = model
        best_params = (hid_dim, learning_rate, batch_size)
        loss_hist = lh
        best_loss = loss

print(
    f"Best hyperparameters are: batch size = {best_params[2]}, hidden layers = {best_params[0]}, lr = {best_params[1]}")

# Load test data
testloader = torch.utils.data.DataLoader(testset, batch_size=32)

# Evaluate the best model on testset.

x_test, _ = next(iter(testloader))
x_test = x_test.to(DEVICE)
x_noise = add_noise(x_test)
x_denoised = best_model(x_noise).detach()

# Plot some examples of original, noisy and denoised inputs.

fig, ax = plt.subplots(32, 3, figsize=(6, 60))
fig.tight_layout()
for i in range(32):
    ax[i][0].imshow(x_test[i, 0].cpu().numpy(), cmap="gray")
    ax[i][1].imshow(x_noise[i, 0].cpu().numpy(), cmap="gray")
    ax[i][2].imshow(x_denoised[i, 0].cpu().numpy(), cmap="gray")

cols = ["Original input", "Noisy", "Denoised"]
for ax, col in zip(ax[0], cols):
    ax.set_title(col)
