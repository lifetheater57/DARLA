import torch.nn as nn
import numpy as np

def genReLUCNN(in_size, out_size, kernel_size, stride):
    # Initialize the CNN
    cnn = nn.Conv2d(in_size, out_size, kernel_size, stride)
    # Initialize the CNN weights with a He normal distribution
    nn.init.kaiming_normal_(cnn.weight, a=0.01, nonlinearity="leaky_relu")
    
    return nn.Sequential(
        cnn,
        nn.BatchNorm2d(out_size),
        nn.LeakyReLU()
    )

def genReLUCNNTranpose(
    in_size, out_size, kernel_size, stride, output_padding=0
):
    # Initialize the transposed CNN
    cnnTransposed = nn.ConvTranspose2d(
        in_size, 
        out_size, 
        kernel_size, 
        stride, 
        output_padding=output_padding
    )
    # Initialize the transposed CNN weights with a He normal distribution
    nn.init.kaiming_normal_(cnnTransposed.weight, a=0.01, nonlinearity="leaky_relu")
    
    return nn.Sequential(
        cnnTransposed,
        nn.BatchNorm2d(out_size),
        nn.LeakyReLU()
    )

def outSizeCNN(
    in_size, kernel_size, stride, n=1, transposed=False, output_padding=0
):
    # Initialize the size list
    size_list = np.zeros((n + 1, 2))
    # Set the first list element to the input size
    size_list[0] = in_size

    for i in np.arange(n) + 1:
        # Set the output size of the current layer
        if transposed:
            size_list[i] = (
                (size_list[i - 1] - 1) * stride + kernel_size + output_padding
            )
        else:
            size_list[i] = np.floor((size_list[i - 1] - kernel_size) / stride + 1)

    return size_list.astype(int)