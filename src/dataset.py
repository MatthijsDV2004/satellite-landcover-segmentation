from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset


def binarize_mask(mask_rgb, threshold=128):
    return np.where(mask_rgb >= threshold, 255, 0).astype(np.uint8)


def rgb_mask_to_class_mask(mask_rgb, color_map):
    mask_rgb = binarize_mask(mask_rgb)
    class_mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)

    for color, class_idx in color_map.items():
        matches = np.all(mask_rgb == np.array(color), axis=-1)
        class_mask[matches] = class_idx

    return class_mask


class DeepGlobeDataset(Dataset):
    def __init__(self, image_paths, color_map, transform=None):
        self.image_paths = image_paths
        self.color_map = color_map
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = img_path.parent / img_path.name.replace("_sat.jpg", "_mask.png")

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_rgb = cv2.imread(str(mask_path))
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        mask = rgb_mask_to_class_mask(mask_rgb, self.color_map)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask


class ProcessedDeepGlobeDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])
        mask = np.load(self.mask_paths[idx])

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask