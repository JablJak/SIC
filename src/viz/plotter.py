import torch
from matplotlib import pyplot as plt


def plot_reconstructions(x_batch: torch.Tensor, x_recon: torch.Tensor, save_path: str | None, show=True):
    """
    TODO: Complete docstring
    Visualize the original and reconstructed images side by side for comparison.

    :param x_batch: A batch of original images. Shape: (batch_size, channels, height, width).
                    The images are assumed to be normalized tensors.
    :param x_recon: A batch of reconstructed images corresponding to the original images. 
                    Shape: (batch_size, channels, height, width).

    :return: None. Displays a matplotlib figure showing the images and reconstructions.
    """

    fig, axes = plt.subplots(6, 5, figsize=(36, 20))

    for row in range(0, 6, 2):
        for col in range(5):
            i = (row//2) * 5 + col
            axes[row, col].imshow(x_batch[i].permute(1, 2, 0).detach().numpy())
            axes[row, col].set_title("Oryginał")
            axes[row, col].axis("off")

            axes[row + 1, col].imshow(x_recon[i].permute(1, 2, 0).detach().numpy())
            axes[row + 1, col].set_title("Rekonstrukcja")
            axes[row + 1, col].axis("off")

    plt.tight_layout()
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)