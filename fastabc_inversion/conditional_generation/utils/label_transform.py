import torch
from skbio.stats.composition import multi_replace, clr, clr_inv


class LabelTransform:
    """
    A callable class to perform One-Hot Encoding.
    It also defines methods to apply Center Log-Ratio (CLR) transformation and its inverse.
    """

    def __init__(self, num_classes, delta):
        """
        Initializes the transform with the required number of classes.

        Args:
            num_classes: The total number of classes (e.g., 10 for MNIST).
            delta: Small constant for multiplicative replacement to avoid zeros.
        """
        self.num_classes = num_classes
        self.delta = delta

    def __call__(self, label):
        """
        Transform label to one-hot encoded tensor.

        :param label: The label to be transformed.
        :return: A PyTorch tensor after applying one-hot encoding.
        """
        # One-Hot Encoding (Output: torch.Tensor)
        one_hot_y = torch.zeros(self.num_classes, dtype=torch.float)
        one_hot_y[label] = 1.0

        return one_hot_y

    def clr_transform(self, one_hot_tensor):
        """
        Applies the CLR transformation to a one-hot encoded tensor.

        one_hot_tensor: A one-hot encoded tensor of shape (N)
                                       or (batch_size, N).

        Returns:
            torch.Tensor: The CLR-transformed tensor.
        """

        # Check if the tensor is a batch or a single sample
        is_batch = one_hot_tensor.dim() > 1

        # 1. Convert to NumPy (scikit-bio input requirement)
        # scikit-bio composition functions prefer 2D arrays (samples x features)
        x_np = one_hot_tensor.detach().cpu().numpy()

        # Reshape for single sample if necessary
        if not is_batch:
            x_np = x_np.reshape(1, self.num_classes)

        # 2. Apply Multiplicative Replacement (Output: numpy.ndarray)
        x_replaced_np = multi_replace(x_np, delta=self.delta)

        # 3. Apply CLR (Output: numpy.ndarray)
        clr_y_np = clr(x_replaced_np)

        # 4. Convert back to PyTorch Tensor
        clr_y_tensor = torch.from_numpy(clr_y_np).float()
        if not is_batch:
            clr_y_tensor = clr_y_tensor.flatten()
        return clr_y_tensor

    def revert_to_label(self, clr_tensor, one_hot=False):
        """
        Applies the inverse CLR transformation to a CLR-transformed tensor.

        clr_tensor: A CLR-transformed tensor, typically of shape (N)
                                       or (batch_size, N).
        one_hot: If True, returns the one-hot encoded vector reverted from CLR.
                    If False, returns the label index with the highest value.

        Returns:
            torch.Tensor: The reconstructed compositional vector(s) where components
                          sum to 1.
        """

        # Check if the tensor is a batch or a single sample
        is_batch = clr_tensor.dim() > 1

        # 1. Convert to NumPy (scikit-bio input requirement)
        # scikit-bio composition functions prefer 2D arrays (samples x features)
        x_np = clr_tensor.detach().cpu().numpy()

        # Reshape for single sample if necessary
        if not is_batch:
            x_np = x_np.reshape(1, self.num_classes)

        # 2. Apply Inverse CLR (Output: numpy.ndarray)
        # This transforms the centered log-ratio vector back to the compositional space.
        compositional_y_np = clr_inv(x_np)

        # Convert back to PyTorch Tensor
        compositional_y_tensor = torch.from_numpy(compositional_y_np).float()
        if not is_batch:
            compositional_y_tensor = compositional_y_tensor.flatten()

        # 3. Check if one_hot output is desired
        if one_hot:
            return compositional_y_tensor
        else:
            labels = self.simplex_vec_to_label(compositional_y_tensor)
            return labels

    def simplex_vec_to_label(self, one_hot_tensor):
        """
        Converts a one-hot encoded tensor back to label indices.

        one_hot_tensor: A one-hot encoded tensor of shape (N)
                                       or (batch_size, N).

        Returns:
            torch.Tensor: The label indices.
        """

        # Check if the tensor is a batch or a single sample
        is_batch = one_hot_tensor.dim() > 1

        # Convert back to PyTorch Tensor
        if is_batch:
            labels = torch.argmax(one_hot_tensor, dim=1)
        else:
            labels = torch.argmax(one_hot_tensor)

        return labels


# Example usage:
if __name__ == "__main__":
    num_classes = 10
    delta = 1e-8
    label_transform = LabelTransform(num_classes=num_classes, delta=delta)

    # Example label
    label = 3

    # One-Hot Encoding
    one_hot = label_transform(label)
    print("One-Hot Encoded:", one_hot)

    # CLR Transformation
    clr_transformed = label_transform.clr_transform(one_hot)
    print("CLR Transformed:", clr_transformed)

    # Inverse CLR Transformation to get back the label
    recovered_label = label_transform.revert_to_label(clr_transformed)
    print("Recovered Label:", recovered_label)
    # Inverse CLR Transformation to get back the one-hot vector
    recovered_one_hot = label_transform.revert_to_label(clr_transformed, one_hot=True)
    print("Recovered One-Hot:", recovered_one_hot)
