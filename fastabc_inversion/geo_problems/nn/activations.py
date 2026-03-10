# -*- coding: utf-8 -*-
"""
Written by Eliane Maalouf (eliane.maalouf@unine.ch), assisted by ChatGPT o1-preview
Definition of Beta-Swish activation function and its constrained version
based on SEARCHING FOR ACTIVATION FUNCTIONS (https://arxiv.org/pdf/1710.05941).
Definition of a constrained PReLU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaSwish(nn.Module):
    def __init__(self, num_parameters=1, init=1.0, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_parameters = num_parameters
        # Initialize beta as a learnable parameter
        self.beta = nn.Parameter(
            torch.empty(num_parameters, **factory_kwargs).fill_(init)
        )

    def forward(self, x):
        dims = [1] * (x.dim() - 2)
        beta = self.beta.view(1, -1, *dims)
        return x * torch.sigmoid(beta * x)


class ConstrainedBetaSwish(BetaSwish):
    def __init__(
        self,
        num_parameters=1,
        init=1.0,
        beta_min=0.1,
        beta_max=None,
        device=None,
        dtype=None,
    ):
        super().__init__(
            num_parameters=num_parameters, init=init, device=device, dtype=dtype
        )
        # Store the constraints
        self.beta_min = beta_min
        self.beta_max = beta_max
        # Create beta_min and beta_max tensors
        factory_kwargs = {"device": self.beta.device, "dtype": self.beta.dtype}
        if beta_min is not None:
            self.register_buffer(
                "beta_min_tensor", torch.tensor(beta_min, **factory_kwargs)
            )
        if beta_max is not None:
            self.register_buffer(
                "beta_max_tensor", torch.tensor(beta_max, **factory_kwargs)
            )

    def forward(self, x):
        # In-place clamp self.beta
        with torch.no_grad():
            if self.beta_min is not None:
                self.beta.data.clamp_(min=self.beta_min)
            if self.beta_max is not None:
                self.beta.data.clamp_(max=self.beta_max)

        return super().forward(x)


class _ConstrainedPReLU(nn.Module):
    # TODO: to delete
    def __init__(
        self, num_parameters=1, init=0.25, max_value=1.0, device=None, dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_parameters = num_parameters
        self.max_value = max_value  # Maximum absolute value of 'weight' to constrain Lipschitz constant
        self.weight = nn.Parameter(
            torch.empty(num_parameters, **factory_kwargs).fill_(init)
        )  # Learnable parameter 'weight'

    def forward(self, x):
        # Constrain 'weight' to be within [-max_value, max_value]
        constrained_weight = torch.clamp(self.weight, -self.max_value, self.max_value)
        # Apply PReLU activation with constrained 'weight'
        return F.prelu(x, constrained_weight)

    def extra_repr(self):
        return "num_parameters={}".format(self.num_parameters)


class ConstrainedPReLU(nn.PReLU):
    def __init__(
        self, num_parameters=1, init=0.25, max_value=1.0, device=None, dtype=None
    ):
        super().__init__(
            num_parameters=num_parameters, init=init, device=device, dtype=dtype
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_parameters = num_parameters
        self.max_value = max_value
        self.register_buffer(
            "max_value_tensor", torch.tensor(max_value, **factory_kwargs)
        )  # Maximum absolute value of 'a' to constrain Lipschitz constant

    def forward(self, x):
        # Constrain 'a' to be within [-max_value, max_value]
        with torch.no_grad():
            self.weight.data.clamp_(min=-self.max_value, max=self.max_value)
        return super().forward(x)


# Test functions
def test_beta_swish_functions():
    # Create a sample input tensor
    x = torch.randn(16, 64, 32, 32)  # Batch size 16, channels 64, height and width 32

    print("Testing BetaSwish Activation Function")
    # Instantiate BetaSwish with per-channel beta parameters
    beta_swish = BetaSwish(num_parameters=64, init=1.0)

    # Forward pass
    output = beta_swish(x)
    print("BetaSwish Output shape:", output.shape)
    print("BetaSwish beta shape:", beta_swish.beta.shape)

    # Verify that beta is a learnable parameter
    assert beta_swish.beta.requires_grad, "Beta parameter should require gradients."

    # Simulate training to see if beta is updated
    optimizer = torch.optim.SGD(beta_swish.parameters(), lr=0.01)
    initial_beta_mean = beta_swish.beta.mean().item()
    print(f"Initial BetaSwish beta mean: {initial_beta_mean:.4f}")

    for i in range(5):
        optimizer.zero_grad()
        output = beta_swish(x)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        current_beta_mean = beta_swish.beta.mean().item()
        print(f"Iteration {i + 1}, BetaSwish beta mean: {current_beta_mean:.4f}")

    print("\nTesting ConstrainedBetaSwish Activation Function")
    # Define beta constraints
    beta_min = 0.5
    beta_max = 2.0

    # Instantiate ConstrainedBetaSwish with per-channel beta parameters
    constrained_beta_swish = ConstrainedBetaSwish(
        num_parameters=64, init=1.0, beta_min=beta_min, beta_max=beta_max
    )

    # Forward pass
    output = constrained_beta_swish(x)
    print("ConstrainedBetaSwish Output shape:", output.shape)
    print("ConstrainedBetaSwish beta shape:", constrained_beta_swish.beta.shape)

    # Verify that beta is a learnable parameter
    assert (
        constrained_beta_swish.beta.requires_grad
    ), "Beta parameter should require gradients."

    # Simulate training to see if beta stays within constraints
    optimizer = torch.optim.SGD(constrained_beta_swish.parameters(), lr=0.1)
    initial_beta_mean = constrained_beta_swish.beta.mean().item()
    print(f"Initial ConstrainedBetaSwish beta mean: {initial_beta_mean:.4f}")

    for i in range(10):
        optimizer.zero_grad()
        output = constrained_beta_swish(x)
        loss = output.mean()
        loss.backward()
        optimizer.step()

        # Get beta values
        beta_values = constrained_beta_swish.beta.data
        beta_mean = beta_values.mean().item()
        beta_min_val = beta_values.min().item()
        beta_max_val = beta_values.max().item()

        # Verify constraints
        beta_below_min = beta_values < beta_min - 1e-6
        beta_above_max = beta_values > beta_max + 1e-6

        print(
            f"Iteration {i + 1}, Beta mean: {beta_mean:.4f}, Beta min: {beta_min_val:.4f}, Beta max: {beta_max_val:.4f}"
        )

        assert (
            not beta_below_min.any()
        ), f"Beta values below beta_min at iteration {i + 1}!"
        assert (
            not beta_above_max.any()
        ), f"Beta values above beta_max at iteration {i + 1}!"

    print(
        "\nConstrainedBetaSwish beta parameters are within the specified constraints after optimization."
    )


def test_constrained_prelu():
    print("\nTesting ConstrainedPReLU Activation Function")
    # Create a sample input tensor
    x = torch.randn(16, 64, 32, 32)  # Batch size 16, channels 64, height and width 32

    # Define constraints
    max_value = 0.5  # Maximum absolute value for 'weight'

    # Instantiate ConstrainedPReLU with per-channel 'weight' parameters
    constrained_prelu = ConstrainedPReLU(
        num_parameters=64, init=0.25, max_value=max_value
    )

    # Forward pass
    output = constrained_prelu(x)
    print("ConstrainedPReLU Output shape:", output.shape)
    print("ConstrainedPReLU a shape:", constrained_prelu.weight.shape)
    print(constrained_prelu.extra_repr())  # method inherited from nn.PReLU

    # Verify that 'weight'  is a learnable parameter
    assert (
        constrained_prelu.weight.requires_grad
    ), "'weight' parameter should require gradients."

    # Simulate training to see if 'a' stays within constraints
    optimizer = torch.optim.SGD(constrained_prelu.parameters(), lr=0.1)
    initial_weight_mean = constrained_prelu.weight.mean().item()
    print(f"Initial ConstrainedPReLU a mean: {initial_weight_mean:.4f}")

    for i in range(10):
        optimizer.zero_grad()
        output = constrained_prelu(x)
        loss = output.mean()
        loss.backward()
        optimizer.step()

        # Get 'weight' values
        weight_values = constrained_prelu.weight.data
        print(weight_values)
        weight_mean = weight_values.mean().item()
        weight_min_val = weight_values.min().item()
        weight_max_val = weight_values.max().item()

        # Apply constraints manually after optimizer step (if not already done in forward)
        """
        with torch.no_grad():
            constrained_prelu.weight.data.clamp_(-max_value, max_value)
        """

        # Verify constraints
        weight_below_min = weight_values < -max_value - 1e-6
        weight_above_max = weight_values > max_value + 1e-6

        print(
            f"Iteration {i+1}, a mean: {weight_mean:.4f}, a min: {weight_min_val:.4f}, a max: {weight_max_val:.4f}"
        )

        assert (
            not weight_below_min.any()
        ), f"'weight'  values below -max_value at iteration {i+1}!"
        assert (
            not weight_above_max.any()
        ), f"'weight'  values above max_value at iteration {i+1}!"

    print(
        "\nConstrainedPReLU 'weight' parameters are within the specified constraints after optimization."
    )


# test functions
if __name__ == "__main__":
    test_beta_swish_functions()
    test_constrained_prelu()
