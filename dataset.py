import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


def read_xray(path):
    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    xray = xray.astype(np.float32) / 255.0
    # shape: (H, W) -> (1, H, W)
    xray = xray.reshape(1, xray.shape[0], xray.shape[1])
    return torch.tensor(xray, dtype=torch.float32)


def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.float32)
    # shape: (H, W) -> (1, H, W)
    mask = mask.reshape(1, mask.shape[0], mask.shape[1])
    return torch.tensor(mask, dtype=torch.float32)


class Knee_Dataset(Dataset):
    def __init__(self, df, has_mask=True):
        self.df = df
        self.has_mask = has_mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df["xrays"].iloc[index]
        image = read_xray(img_path)

        result = {'image': image}

        if self.has_mask:
            mask_path = self.df["masks"].iloc[index]
            mask = read_mask(mask_path)
            result["mask"] = mask
        else:
            result["name"] = os.path.basename(img_path)

        return result
