import torch
import torch.nn as nn
import numpy as np

from utils.model import outSizeCNN, genReLUCNN, genReLUCNNTranpose


class Model(nn.Module):
    def __init__(self, shape):
        super().__init__()

        # Splitting shape
        obs_channels, obs_height, obs_width = shape

        # Initializing constant params
        kernel = 4
        stride = 2
        filters = [32, 32, 64, 64]

        # Computing the dims required by the flattening and unflattening ops
        in_dims = np.array([obs_height, obs_width])
        out_dims = outSizeCNN(in_dims, kernel, stride, len(filters))
        flattened_dims = filters[-1] * out_dims[-1, 0] * out_dims[-1, 1]

        # Creating the encoder
        CNN_encoder = nn.Sequential()

        for i in range(len(filters)):
            in_channels = filters[i - 1] if i > 0 else obs_channels
            out_channels = filters[i]

            module = genReLUCNN(in_channels, out_channels, kernel, stride)
            module_name = "enc_conv_relu" + str(i)

            CNN_encoder.add_module(module_name, module)

        self.encoder = nn.Sequential(
            CNN_encoder, nn.Flatten(), nn.Linear(flattened_dims, 128)
        )

        # Creating the decoder
        CNN_decoder = nn.Sequential()

        for i in reversed(range(len(filters))):
            in_channels = filters[i]
            out_channels = filters[i - 1] if i > 0 else obs_channels

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

        self.decoder = nn.Sequential(
            nn.Linear(128, flattened_dims),
            nn.Unflatten(
                1, (filters[-1], int(out_dims[-1, 0]), int(out_dims[-1, 1]))
            ),
            CNN_decoder,
        )

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)