import torch


def get_boundary_mask(h, w, device, mode="nhwc", value=0.2):
    if mode == "nhwc":
        mask = torch.zeros((1, h, w, 1), device=device)
        mask[:, 0:, :, :] = value
        mask[:, -1:, :, :] = value
        mask[:, :, 0:, :] = value
        mask[:, :, -1:, :] = value
    if mode == "nchw":
        mask = torch.zeros((1, 1, h, w), device=device)
        mask[:, :, 0:, :] = value
        mask[:, :, -1:, :] = value
        mask[:, :, :, 0] = value
        mask[:, :, :, -1] = value
    return mask