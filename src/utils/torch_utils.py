import torch


def get_boundary_mask(h, w, device, mode="nhwc"):
    if mode == "nhwc":
        mask = torch.zeros((1, h, w, 1), device=device)
        mask[:, 0:, :, :] = 1.0
        mask[:, -1:, :, :] = 1.0
        mask[:, :, 0:, :] = 1.0
        mask[:, :, -1:, :] = 1.0
    if mode == "nchw":
        mask = torch.zeros((1, 1, h, w), device=device)
        mask[:, :, 0:, :] = 1.0
        mask[:, :, -1:, :] = 1.0
        mask[:, :, :, 0] = 1.0
        mask[:, :, :, -1] = 1.0
    return mask