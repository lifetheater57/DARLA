import torch
import torch.nn as nn
import numpy as np

from utils.model import outSizeCNN, genReLUCNN, genReLUCNNTranpose


class Model(nn.Module):
    def __init__(self, shape, latent_dim=32):
        super().__init__()

        # Splitting shape
        obs_channels, obs_height, obs_width = shape

        # Initializing constant params
        kernel = 4
        stride = 2
        filters = [32, 64, 128, 256, 512]

        # Computing the dims required by the flattening and unflattening ops
        in_dims = np.array([obs_height, obs_width])
        out_dims = outSizeCNN(in_dims, kernel, stride, len(filters))
        flattened_dims = filters[-1] * out_dims[-1, 0] * out_dims[-1, 1]

        # Creation of the encoder's CNN
        CNN_encoder = nn.Sequential()
        for i in range(len(filters)):
            in_channels = filters[i - 1] if i > 0 else obs_channels
            out_channels = filters[i]

            module = genReLUCNN(in_channels, out_channels, kernel, stride)
            module_name = "enc_conv_relu" + str(i)

            CNN_encoder.add_module(module_name, module)

        # Initialization of the layer on top of the CNN of the encoder 
        # and its weights and biases
        encoder_linear_layer = nn.Linear(flattened_dims, 256)
        nn.init.kaiming_normal_(encoder_linear_layer.weight, a=0.01, nonlinearity="leaky_relu")

        # Creation of the encoder
        self.encoder = nn.Sequential(
            CNN_encoder, 
            nn.Flatten(), 
            encoder_linear_layer,
            nn.BatchNorm1d(256),
            nn.LeakyReLU()
        )

        # Creation of the latent space mean and variance layers
        self.mu = nn.Sequential(nn.Linear(256, latent_dim))
        self.log_var = nn.Sequential(nn.Linear(256, latent_dim))

        # Creation of the decoder's CNN
        CNN_decoder = nn.Sequential()
        for i in reversed(range(len(filters))):
            in_channels = filters[i]
            out_channels = filters[i - 1] if i > 0 else obs_channels#2 * obs_channels

            out_size = outSizeCNN(
                out_dims[i + 1], kernel, stride, transposed=True
            )[1]
            output_padding = tuple(out_dims[i] - out_size)

            module = genReLUCNNTranpose(
                in_channels,
                out_channels,
                kernel,
                stride,
                output_padding=output_padding
            )
            module_name = "dec_relu_conv" + str(len(filters) - i - 1)

            CNN_decoder.add_module(module_name, module)

        # Initialization of the layer on top of the CNN of the decoder 
        # and its weights and biases
        decoder_linear_layer = nn.Linear(latent_dim, 256)
        nn.init.kaiming_normal_(decoder_linear_layer.weight, a=0.01, nonlinearity="leaky_relu")
        
        # Creation of the decoder 
        self.decoder = nn.Sequential(
            decoder_linear_layer,
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, flattened_dims),
            nn.Unflatten(1, (filters[-1], int(out_dims[-1, 0]), int(out_dims[-1, 1]))),
            CNN_decoder,
        )

    def forward(self, x):
        # Encode the example
        x = self.encoder(x)        
        # Get mean and variance of the latent variables for the example
        mu = self.mu(x)
        log_var = self.log_var(x)
        # Sample from the latent space
        z = self.sample_latent_space(mu, log_var)        
        # Decode the sample
        x_hat = self.decode(z)

        return x_hat, mu, log_var

    def sample_latent_space(self, mu, log_var):
        return mu + torch.mul(torch.exp(log_var / 2.0), torch.randn_like(log_var))

    def encode(self, x):
        # Encode the example
        x = self.encoder(x)
        # Get mean and variance of the latent variables from the encoded example
        mu = self.mu(x)
        log_var = self.log_var(x)
        # Sample from the latent space
        z = self.sample_latent_space(mu, log_var)
        
        return z

    def decode(self, z):
        # Decode the sample
        decoded = self.decoder(z)
        """# Get mean and variance of the output values from the decoded sample
        mus = decoded[:, 0::2, :, :]
        log_vars = decoded[:, 1::2, :, :]
        # Sample from the parameters
        sampled = self.sample_latent_space(mus, log_vars)"""
        
        return decoded#sampled