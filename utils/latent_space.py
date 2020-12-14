import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

#%%


def getLatentSpaceStats(encoded_data, return_dataframe=False):
    # Encode the data in latent space
    # encoded_data = encoder.encode(data)
    columns = ["dim" + str(i) for i in range(encoded_data.shape[1])]
    df = pd.DataFrame(data=encoded_data, columns=columns)

    # Describe the latent variables
    if return_dataframe:
        return df
    else:
        return df.describe()


def get_bounds(loader, encoder, latent=32, common_bounds=False):
    """
    get bounds from the whole dataset
    """
    bounds = np.zeros((latent, 2))
    for i, elem in enumerate(loader):
        encoded_data = encoder.encode(elem)
        stats = getLatentSpaceStats(
            encoded_data.data.cpu().numpy(), return_dataframe=False
        )
        mins = stats.loc["min"]
        maxs = stats.loc["max"]

        bounds = np.array(
            [
                np.array([mins, bounds[:, 0]]).min(axis=0),
                np.array([maxs, bounds[:, 1]]).max(axis=0),
            ]
        ).transpose(1, 0)

    if common_bounds:
        return np.array([bounds[0].min(), bounds[1].max()])
    else:
        return bounds


#%%
def norm(decoded_samples):
    decoded_samples = decoded_samples - decoded_samples.min()
    decoded_samples = decoded_samples / decoded_samples.max()
    return decoded_samples


def traversals(model, shape, dimensions, bounds, num_samples, state):
    # Initializing size attributes and figure
    n_dims = len(dimensions)
    width, height, num_channels = shape[2], shape[1], shape[0]
    figure = np.zeros((width * n_dims, height * num_samples, num_channels))

    assert bounds.shape == (2) or bounds.shape == (n_dims, 2), (
        "cannot broadcast bounds size "
        + str(bounds.shape)
        + " with dimensions size "
        + str(len(dimensions))
    )

    # Creating values for the sampling
    if len(bounds.shape) == 1:
        min_val = bounds[0]
        max_val = bounds[1]
        columns = [np.linspace(min_val, max_val, num_samples)[::-1]] * n_dims
    else:
        columns = []
        for min_val, max_val in bounds:
            columns.append(np.linspace(min_val, max_val, num_samples)[::-1])

    # Generating images
    for i, dim in enumerate(dimensions):
        z_samples = torch.stack([state.detach().clone()] * len(columns[i]))
        for j, value in enumerate(columns[i]):
            z_samples[j][dim] = value

        # Decode samples
        decoded_samples = model.decoder(z_samples)
        # Transpose into a shape usable by matplotlib
        print(decoded_samples.shape)
        decoded_samples = np.hstack(decoded_samples.detach().numpy()).transpose(2, 1, 0)
        print(decoded_samples.shape)
        # Reshape into a figure column
        # decoded_samples = decoded_samples.reshape(
        #    (width, height * num_samples, num_channels)
        # )
        # print(decoded_samples.shape)
        # Assign to the colomn in the figure
        figure[(i * width) : ((i + 1) * width)] = (
            np.clip(decoded_samples, 0, 1) * 255
        ).astype(int)

    plt.figure(figsize=(12, 12))
    start_range = width // 2
    end_range = num_samples * width + start_range + 1
    pixel_range = np.arange(start_range, end_range, width)
    # sample_range_x = np.round(grid_x, 1)
    # sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range)  # , sample_range_x)
    plt.yticks(pixel_range)  # , sample_range_y)
    plt.xlabel("z - dim 1")
    plt.ylabel("z - dim 2")
    # matplotlib.pyplot.imshow() needs a 2D array, or a 3D array with the third dimension being of shape 3 or 4!
    # So reshape if necessary
    fig_shape = np.shape(figure)
    if fig_shape[2] == 1:
        figure = figure.reshape((fig_shape[0], fig_shape[1]))
    # Show image
    plt.imshow(figure)
    plt.show()

