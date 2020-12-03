import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, shape):
        super(Model, self).__init__()

        # Splitting shape
        obs_channels, obs_height, obs_width = shape

        # Initializing constant params
        kernel = 4
        stride = 2
        filters = [32, 32, 64, 64]
        latent_dim = 32

        # Computing the dims required by the flattening and unflattening ops
        in_dims = np.array([obs_height, obs_width])
        out_dims = self.outSize(in_dims, kernel, stride, len(filters))
        flattened_dims = filters[-1] * out_dims[0] * out_dims[1]

        # Creating the encoder
        CNN_encoder = nn.Sequential()

        for i in range(len(filters)):
            in_size = filters[i - 1] if i > 0 else obs_channels
            out_size = filters[i]

            module = self.genReLUCNN(in_size, out_size, kernel, stride)
            module_name = "enc_conv_relu" + str(i)

            CNN_encoder.add_module(module_name, module)

        self.encoder = nn.Sequential(
            CNN_encoder,
            nn.Flatten(),
            nn.Linear(flattened_dims, 256)
        )

        self.mu = nn.Sequential(
            nn.Linear(256, latent_dim)
        )

        self.log_var = nn.Sequential(
            nn.Linear(256, latent_dim)
        )

        # Creating the decoder
        CNN_decoder = nn.Sequential()

        for i in reversed(range(len(filters))):
            in_size = filters[i]
            out_size = filters[i - 1] if i > 0 else 2 * obs_channels

            module = self.genReLUCNNTranpose(in_size, out_size, kernel, stride)
            module_name = "dec_relu_conv" + str(len(filters) - i - 1)

            CNN_decoder.add_module(module_name, module)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256)
            nn.Linear(256, flattened_dims),
            nn.Unflatten(1, (int(filters[-1]), out_dims[0], out_dims[1])),
            CNN_decoder,
        )

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)

        z = mu + torch.mul(torch.exp(log_var / 2.0), torch.randn_like(log_var))
        x_hat = self.decoder(z)

        return x_hat, mu, log_var

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = mu + torch.mul(torch.exp(log_var / 2.0), torch.randn_like(log_var))
        return z

    def decode(self, z):
        return self.decoder(z)

    def genReLUCNN(self, in_size, out_size, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size, stride), 
            nn.ReLU()
        )

    def genReLUCNNTranpose(self, in_size, out_size, kernel_size, stride):
        return nn.Sequential(
            nn.ReLU(), 
            nn.ConvTranspose2d(in_size, out_size, kernel_size, stride)
        )

    def outSize(self, in_size, kernel_size, stride, n=1):
        size = in_size

        for _ in range(n):
            size = np.round((size - kernel_size) / stride + 1).astype(int)

        return size

