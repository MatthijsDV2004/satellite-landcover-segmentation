from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def binarize_mask(mask_rgb, threshold=128):
    return np.where(mask_rgb >= threshold, 255, 0).astype(np.uint8)


def rgb_mask_to_class_mask(mask_rgb, color_map):
    mask_rgb = binarize_mask(mask_rgb)
    class_mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)

    for color, class_idx in color_map.items():
        matches = np.all(mask_rgb == np.array(color), axis=-1)
        class_mask[matches] = class_idx

    return class_mask


DATA_DIR = Path("data/raw/deepglobe")
TRAIN_DIR = DATA_DIR / "train"
CLASS_CSV = DATA_DIR / "class_dict.csv"

OUT_IMG_DIR = Path("data/processed/train_images")
OUT_MASK_DIR = Path("data/processed/train_masks")

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

class_df = pd.read_csv(CLASS_CSV)

color_map = {}
for idx, row in class_df.iterrows():
    color_map[(row["r"], row["g"], row["b"])] = idx

image_paths = sorted(TRAIN_DIR.glob("*_sat.jpg"))

target_size = (256, 256)

for img_path in tqdm(image_paths):
    mask_path = TRAIN_DIR / img_path.name.replace("_sat.jpg", "_mask.png")

    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    mask_rgb = cv2.imread(str(mask_path))
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
    class_mask = rgb_mask_to_class_mask(mask_rgb, color_map)
    class_mask = cv2.resize(class_mask, target_size, interpolation=cv2.INTER_NEAREST)

    out_img = OUT_IMG_DIR / img_path.name.replace(".jpg", ".npy")
    out_mask = OUT_MASK_DIR / mask_path.name.replace(".png", ".npy")

    np.save(out_img, image)
    np.save(out_mask, class_mask)

print("Done.")