import torch
from piqa import MS_SSIM
from pytorch_msssim import ms_ssim


# TODO: Docstrings
class L1SSIM(torch.nn.Module):
    """
    A PyTorch module that combines Mean Squared Error (MSE) loss and Structural Similarity Index (SSIM) loss
    for image reconstruction tasks. The combination is controlled by a weighting parameter alpha, which determines
    the trade-off between MSE and SSIM losses.
    """

    def __init__(self, alpha=0):
        """
        Initialize the MSE_SSIM module.

        Args:
            alpha (float): Weighting factor between MSE (1 - alpha) and SSIM (alpha). Defaults to 0.75.
        """
        super(L1SSIM, self).__init__()
        self.l1 = torch.nn.L1Loss()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined MSE and SSIM loss.

        Args:
            pred (torch.Tensor): Predicted image tensor with shape (N, C, H, W).
            target (torch.Tensor): Target image tensor with shape (N, C, H, W).

        Returns:
            torch.Tensor: The calculated loss value.
        """
        ms_ssim_loss = ms_ssim(pred, target, data_range=1.0) if self.alpha != 0 else 0
        return self.alpha * (1 - ms_ssim_loss) + (1 - self.alpha) * self.l1(pred, target)
