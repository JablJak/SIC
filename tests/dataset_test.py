import os
import tempfile

import torch
from PIL import Image
from src.data.base_dataset import BaseDataset


def test_basedataset_getitem():
    with tempfile.TemporaryDirectory() as tmpdir:
        class_dir = os.path.join(tmpdir, "class0")
        os.makedirs(class_dir)

        img_path = os.path.join(class_dir, "test.png")
        img = Image.new("RGB", (1024, 1024), color=(255, 0, 0))
        img.save(img_path)

        patch_sizes = [(256, 384), (384, 256)]
        dataset = BaseDataset(root_dir=tmpdir, patch_sizes=patch_sizes)

        patch_groups, target = dataset[0]

        assert isinstance(target, int)
        assert target == 0  #

        assert isinstance(patch_groups, list)
        assert len(patch_groups) == 2

        # first group: (256x384)
        g0 = patch_groups[0]
        assert isinstance(g0, torch.Tensor)
        assert g0.ndim == 4  # [N, C, H, W]
        assert g0.shape[1:] == (3, 256, 384)
        assert g0.shape[0] == 12

        # second group: (384x256)
        g1 = patch_groups[1]
        assert g1.ndim == 4
        assert g1.shape[1:] == (3, 384, 256)
        assert g1.shape[0] == 12

        total_patches = g0.shape[0] + g1.shape[0]
        assert total_patches == 24