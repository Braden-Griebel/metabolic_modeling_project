"""
Defining a VAE Model Using Pytorch
"""

# Standard Librar Imports
from typing import List

# External Imports
import torch
import torch.nn as nn

# Local Imports


# VAE Class
class MetabolicVAE(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dims: List[int], latent_dim: int, device
    ) -> None:
        """
        Create the VAE Object

        Args:
            input_dim: Input dimension, number of features of the dataset to encode
            hidden_dims: List of sizes of the hidden dimensions
            latent_dim: Size of the latent dimension
            device: Torch device being used

        Returns:
            None
        """
        super(MetabolicVAE, self).__init__()
        self.device = device

        # Encoding
        # Build up encoder with different hidden_dims
        # Initial Layer
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.LeakyReLU()]
        # Build up the hidden Layers
        lag_dim = hidden_dims[0]
        for dim in hidden_dims[1:]:
            layers += [nn.Linear(lag_dim, dim), nn.LeakyReLU()]
            lag_dim = dim
        # Build up to the encoder
        self.encoder = nn.Sequential(*layers)
        # Generate the Mean and Variance in the Latent Space
        self.mean_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)

        # Create Decoder
        layers = [nn.Linear(latent_dim, hidden_dims[-1]), nn.LeakyReLU()]
        lag_dim = hidden_dims[-1]
        for dim in hidden_dims[-2::-1]:
            layers += [nn.Linear(lag_dim, dim), nn.LeakyReLU()]
        layers += [
            nn.Linear(hidden_dims[0], input_dim),
            nn.ReLU(),
        ]  # This is assuming all the fluxes are positive, so split forward and reverse rxns
        self.decoder = nn.Sequential(*layers)

    def encode(self, x):
        """Encode x to the latent dim mean and logvar"""
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """Use reparameterization trick to generate a random sample from the mean and logvar"""
        var = torch.exp(logvar)
        eps = torch.randn_like(var)
        return mean + var * eps

    def decode(self, x):
        """
        Decode from the latent representation
        """
        return self.decoder(x)

    def forward(self, x):
        """Run the forward pass"""
        mean, logvar = self.encode(x)
        latent_sample = self.reparameterize(mean, logvar)
        x_recon = self.decode(latent_sample)
        return (
            x_recon,
            mean,
            logvar,
        )  # x_recon used for reconstruction error, mean/logvar for KL divergence term


def loss_function(x, x_recon, mean, logvar):
    """Calculate the Loss of the VAE"""
    reproduction_loss = nn.functional.mse_loss(x_recon, x)
    kl_loss = -0.5 * torch.sum((1 + logvar - mean.power(2) - torch.exp(logvar)))
    return reproduction_loss + kl_loss


def train(model, optimizer, train_loader, x_dim, epochs, batch_size, device):
    model.train(True)  # put model in training mode so that gradient tracking is on
    overall_loss = 0
    for _ in range(epochs):
        overall_loss = 0
        for _, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, x_dim).to(device)

            optimizer.zero_grad()  # Clear the accumulated gradients

            x_recon, mean, logvar = model(x)
            loss = loss_function(x, x_recon, mean, logvar)

            overall_loss += loss.item()

            loss.backward()  # Accumulate grads for this batch
            optimizer.step()  # Step the optimizer using the batches gradients

    return overall_loss
