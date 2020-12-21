from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch


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


def get_bounds(loader, vae, latent=32, common_bounds=False):
    """
    get bounds from the whole dataset
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bounds = np.zeros((latent, 2))
    for i, elem in enumerate(loader):
        elem = elem.to(device)
        encoded_data = vae.encode(elem)
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


@torch.no_grad()
def traversals(model, shape, dimensions, bounds, num_samples, state, filepath):
    
    # Initializing size attributes and figure
    n_dims = len(dimensions)
    width, height, num_channels = shape[-1], shape[-2], shape[-3]
    figure = np.zeros((height * num_samples, width * n_dims, num_channels))

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
        z_samples = torch.stack([state.clone()] * len(columns[i]))
        for j, value in enumerate(columns[i]):
            z_samples[j][dim] = value

        # Decode samples
        decoded_samples = model.decode(z_samples)
        # Transpose into a shape usable by matplotlib
        decoded_samples = decoded_samples.cpu().numpy().transpose(0, 2, 3, 1)
        # Reshape into a figure column
        decoded_samples = np.vstack(decoded_samples)
        # Assign to the colomn in the figure
        figure[:, (i * width) : ((i + 1) * width)] = (np.clip(decoded_samples, 0, 1) * 255).astype(int)
        
    plt.figure(figsize=(128, 48))
    start_x = width // 2
    end_x = n_dims * width + start_x
    start_y = height // 2
    end_y = num_samples * height + start_y
    x_range = np.linspace(start_x, end_x, n_dims + 1)
    y_range = np.linspace(start_y, end_y, num_samples + 1)
    plt.xticks(x_range, np.arange(len(x_range)))
    plt.yticks(y_range, np.arange(len(y_range)))
    plt.xlabel("Dimension")
    plt.ylabel("Image")
    # matplotlib.pyplot.imshow() needs a 2D array, or a 3D array with the third dimension being of shape 3 or 4!
    # So reshape if necessary
    fig_shape = np.shape(figure)
    if fig_shape[2] == 1:
        figure = figure.reshape((fig_shape[0], fig_shape[1]))
    # Show image
    plt.imshow(figure)
    plt.savefig(filepath)
    print("Traversal saved at: " + str(filepath))

