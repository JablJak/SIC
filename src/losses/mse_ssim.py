import torch
from piqa import MS_SSIM
from piqa.ssim import ms_ssim
from piqa.utils.functional import gaussian_kernel


class MSESSIM(torch.nn.Module):
    """
    A PyTorch module that combines Mean Squared Error (MSE) loss and Structural Similarity Index (SSIM) loss
    for image reconstruction tasks. The combination is controlled by a weighting parameter alpha, which determines
    the trade-off between MSE and SSIM losses.
    """

    def __init__(self, alpha=0.75):
        """
        Initialize the MSE_SSIM module.

        Args:
            alpha (float): Weighting factor between MSE (1 - alpha) and SSIM (alpha). Defaults to 0.75.
        """
        super(MSESSIM, self).__init__()
        self.l2 = torch.nn.MSELoss()
        self.activation = torch.nn.Sigmoid()
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
        pred = self.activation(pred)
        target = self.activation(target)
        kernel = gaussian_kernel(7).repeat(3, 1, 1).to(pred.device)
        ssim_loss = ms_ssim(pred, target, kernel, MS_SSIM.WEIGHTS.to(pred.device)).mean()
        return self.alpha * (1 - ssim_loss) + (1 - self.alpha) * self.l2(pred, target)
