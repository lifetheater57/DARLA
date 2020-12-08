import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, Union
from torch import Tensor
from torch import Size


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
        out_dims = self.outSize(in_dims, kernel, stride, len(filters))
        flattened_dims = filters[-1] * out_dims[-1, 0] * out_dims[-1, 1]

        # Creating the encoder
        CNN_encoder = nn.Sequential()

        for i in range(len(filters)):
            in_channels = filters[i - 1] if i > 0 else obs_channels
            out_channels = filters[i]

            module = self.genReLUCNN(in_channels, out_channels, kernel, stride)
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

            out_size = self.outSizeCNN(
                out_dims[-(i + 1)], kernel, stride, transposed=True
            )
            output_padding = out_dims[-(i + 2)] - out_size

            module = self.genReLUCNNTranpose(
                in_channels, out_channels, kernel, stride, output_padding=output_padding
            )
            module_name = "dec_relu_conv" + str(len(filters) - i - 1)

            CNN_decoder.add_module(module_name, module)

        self.decoder = nn.Sequential(
            nn.Linear(128, flattened_dims),
            nn.Unflatten(1, (int(filters[-1]), int(out_dims[0]), int(out_dims[1]))),
            CNN_decoder,
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

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
                in_size,
                out_size,
                kernel_size,
                stride,
                output_padding=output_padding,
                padding_mode="duplicate",
            ),
        )

    def outSizeCNN(
        self, in_size, kernel_size, stride, n=1, transposed=False, output_padding=0
    ):
        size_list = np.zeros((n + 1, 1))
        size_list[0] = in_size

        for i in np.arange(n) + 1:
            if transposed:
                size_list[i] = (size[i - 1] - 1) * stride + kernel_size + output_padding
            else:
                size_list[i] = np.floor(
                    (size[i - 1] - kernel_size) / stride + 1
                ).astype(int)

        return size
