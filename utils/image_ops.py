import torch
from torch import Tensor

# Keep this internal for now!
def _gradient(x: Tensor) -> Tensor:
    """
    Compute 2D gradient as a 2xHxW tensor (channels: dx, dy).
    Pads the borders to retain the original shape.
    """
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)

    gx[:, 1:] = x[:, 1:] - x[:, :-1]
    gy[1:, :] = x[1:, :] - x[:-1, :]

    return torch.stack((gx, gy))