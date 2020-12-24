import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    def __init__(self, shape, latent_dim=32):
        super().__init__()

        # Splitting shape
        obs_channels, obs_height, obs_width = shape

        # Initializing constant params
        kernel = 4
        stride = 2
        filters = [32, 32, 64, 64]

        # Computing the dims required by the flattening and unflattening ops
        in_dims = np.array([obs_height, obs_width])
        out_dims = self.outSizeCNN(in_dims, kernel, stride, len(filters))
        flattened_dims = filters[-1] * out_dims[-1, 0] * out_dims[-1, 1]

        # Creating the encoder
        CNN_encoder = nn.Sequential()

        for i in range(len(filters)):
            in_channels = filters[i - 1] if i > 0 else (2 * obs_channels)
            out_channels = filters[i]

            module = self.genReLUCNN(in_channels, out_channels, kernel, stride)
            module_name = "enc_conv_relu" + str(i)

            CNN_encoder.add_module(module_name, module)

        self.encoder = nn.Sequential(
            CNN_encoder, nn.Flatten(), nn.Linear(flattened_dims, 256)
        )

        self.mu = nn.Sequential(nn.Linear(256, latent_dim))

        self.log_var = nn.Sequential(nn.Linear(256, latent_dim))

        # Creating the decoder
        CNN_decoder = nn.Sequential()

        for i in reversed(range(len(filters))):
            in_channels = filters[i]
            out_channels = filters[i - 1] if i > 0 else 2 * obs_channels

            out_size = self.outSizeCNN(
                out_dims[i + 1], kernel, stride, transposed=True
            )[1]
            output_padding = tuple(out_dims[i] - out_size)

            module = self.genReLUCNNTranpose(
                in_channels, out_channels, kernel, stride, output_padding=output_padding
            )
            module_name = "dec_relu_conv" + str(len(filters) - i - 1)

            CNN_decoder.add_module(module_name, module)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Linear(256, flattened_dims),
            nn.Unflatten(1, (filters[-1], int(out_dims[-1, 0]), int(out_dims[-1, 1]))),
            CNN_decoder,
        )

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)

        z = mu + torch.mul(torch.exp(log_var / 2.0), torch.randn_like(log_var))
        x_hat = self.decode(z)

        return x_hat, mu, log_var

    def sample_latent_space(self, mu, log_var):
        return mu + torch.mul(torch.exp(log_var / 2.0), torch.randn_like(log_var))

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self.sample_latent_space(mu, log_var)
        return z

    def decode(self, z):
        decoded = self.decoder(z)
        mus = decoded[:, 0::2, :, :]
        log_vars = decoded[:, 1::2, :, :]
        sampled = self.sample_latent_space(mus, log_vars)
        return sampled

    def represent(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)

        return [mu, log_var]

    def genReLUCNN(self, in_size, out_size, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size, stride), nn.ReLU()
        )

    def genReLUCNNTranpose(
        self, in_size, out_size, kernel_size, stride, output_padding=0
    ):
        return nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_size, out_size, kernel_size, stride, output_padding=output_padding
            ),
        )

    def outSizeCNN(
        self, in_size, kernel_size, stride, n=1, transposed=False, output_padding=0
    ):
        size_list = np.zeros((n + 1, 2))
        size_list[0] = in_size

        for i in np.arange(n) + 1:
            if transposed:
                size_list[i] = (
                    (size_list[i - 1] - 1) * stride + kernel_size + output_padding
                )
            else:
                size_list[i] = np.floor((size_list[i - 1] - kernel_size) / stride + 1)

        return size_list.astype(int)

