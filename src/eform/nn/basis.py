import torch


def bessel(x, start=0.0, end=1.0, num_basis=8, eps=1e-5):
    """Expand scalar features into (radial) Bessel basis function values.
    """
    x = x[..., None] - start + eps
    c = end - start
    n = torch.arange(1, num_basis+1, dtype=x.dtype, device=x.device)
    return ((2/c)**0.5) * torch.sin(n*torch.pi*x / c) / x
    