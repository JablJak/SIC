import torch
from fontTools.misc.bezierTools import epsilon
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

        self.mse = torch.nn.MSELoss()
        if alpha != 0:
            self.MS_SSIM = MS_SSIM()
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
        self.mse.to(pred.device)
        self.mse.to(pred.device)
        epsilon = 0.001
        mse_scale = 1
        # mse_scale = 255 ** 2
        if pred.min() < (0 - epsilon) or pred.max() > (1 +  epsilon):
            pred = self.activation(pred)
        if target.min() < (0 - epsilon) or target.max() > (1 + epsilon):
            target = self.activation(target)
        mse_loss = self.mse(pred, target) * mse_scale
        ms_ssim_loss = self.MS_SSIM(pred, target) if self.alpha != 0 else 0
        return self.alpha * (1 - ms_ssim_loss) + (1 - self.alpha) * mse_loss
