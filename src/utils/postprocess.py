import torch


def denormalize(x: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    """
    Denormalize the input tensor. The input tensor is normalized by the following formula:
    x = (x - mean) / std
    If the input tensor is 3D, it is assumed to be a single image and is unsqueezed to 4D.
    Thanks to unsqueezing, the function can be used for both single and batch inputs.

    Args:
    x: torch.Tensor, input tensor
    mean: list, mean values used for normalization
    std: list, standard deviation values used for normalization

    Returns:
    torch.Tensor, denormalized input tensor
    """

    if x.ndim == 3:
        x = x.unsqueeze(0)

    mean = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, -1, 1, 1)

    return x * std + mean