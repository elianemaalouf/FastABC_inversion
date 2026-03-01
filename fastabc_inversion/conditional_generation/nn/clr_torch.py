"""
Module implementing Centered Log-Ratio (CLR) transformation using PyTorch tensors in order to
attach to computational graph for backpropagation.
This module defines a CLR class that can be used as a layer in neural networks.
"""

import torch
import torch.nn as nn


class CLR(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, compositional_vec):
        """
        Processes the given compositional vector to compute the forward pass in the model.

        This function defines the forward pass for the model using the provided
        input data, typically represented as a compositional vector.

        :param compositional_vec: The input vector representing compositional
            data for the forward computation.
        :return: The result of the forward pass computation.
        """

        # Add epsilon to avoid log(0). This addition does not preserve the sum-to-one property (sum to 0 in clr space),
        # but the difference is negligible.
        compositional_vec = compositional_vec + self.epsilon

        # Compute the geometric mean
        log_vec = torch.log(compositional_vec)
        mean_log = torch.mean(log_vec, dim=-1, keepdim=True)

        # Compute CLR transformation
        clr_vec = log_vec - mean_log

        return clr_vec


# test code
if __name__ == "__main__":
    clr_layer = CLR()

    # Test with a batch of compositional vectors
    compositional_data = torch.tensor(
        [[0.0, 0.0, 1.0], [0.1, 0.1, 0.8]], dtype=torch.float32
    )

    clr_output = clr_layer(compositional_data)
    print("CLR Output:\n", clr_output)
