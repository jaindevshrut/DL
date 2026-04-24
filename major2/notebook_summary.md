## Cell 1 (markdown)
# Comparative Study of CNN-based vs Vision Transformer-based Models for Semantic Segmentation on Urban Street Scenes

This notebook is a complete, Kaggle-ready deep learning project for **semantic segmentation** with two model families:

- **CNN baseline:** U-Net with a pretrained ResNet encoder
- **Transformer baseline:** SegFormer-B2 with pretrained weights when available

The notebook is written to run **end-to-end on Kaggle (GPU T4)** and includes:

- dataset discovery for **Cityscapes**
- a **synthetic urban-scene fallback** when Cityscapes is unavailable
- preprocessing, augmentations, and data visualization
- training, validation, checkpointing, and resume support
- IoU and Dice metrics
- prediction visualization and overlays
- explainability with **Grad-CAM** for U-Net and **attention maps** for SegFormer
- training and inference time analysis
- final quantitative comparison tables and plots


## Cell 2 (markdown)
## Notebook Notes

- **Primary dataset path:** `/kaggle/input/cityscapes`
- **Input resolution:** `512 x 512`
- **Default training setup:** `5 epochs` on a **subset** of Cityscapes for Kaggle-safe execution
- **Scaling for HPC:** increase `CFG.epochs` to `30-50`
- **Robustness:** if the dataset path or pretrained weights are missing, the notebook falls back gracefully instead of crashing

The code is modular, presentation-ready, and designed for a college project submission.


## Cell 3 (code)
```python
import gc
import json
import math
import os
import random
import time
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from PIL import Image, ImageDraw
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models import ResNet34_Weights, ResNet50_Weights
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
from transformers import SegformerConfig, SegformerForSemanticSegmentation

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["legend.fontsize"] = 11

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

seed_everything(42)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

@dataclass
class Config:
    dataset_dir: str = "/kaggle/input/datasets/shuvoalok/cityscapes"
    image_size: int = 512
    num_classes: int = 21
    ignore_index: int = 255
    encoder_name: str = "resnet34"
    batch_size: int = 4
    segformer_batch_size: int = 2
    epochs: int = 5
    hpc_recommended_epochs: int = 30
    max_train_samples: int = 700
    max_val_samples: int = 140
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = min(4, os.cpu_count() or 2)
    resume_training: bool = True
    seed: int = 42
    train_images_dir: str = field(init=False)
    train_labels_dir: str = field(init=False)
    val_images_dir: str = field(init=False)
    val_labels_dir: str = field(init=False)

    def __post_init__(self) -> None:
        self.train_images_dir = os.path.join(self.dataset_dir, "train", "img")
        self.train_labels_dir = os.path.join(self.dataset_dir, "train", "label")
        self.val_images_dir = os.path.join(self.dataset_dir, "val", "img")
        self.val_labels_dir = os.path.join(self.dataset_dir, "val", "label")

CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"
IS_KAGGLE = Path("/kaggle").exists()
WORKING_DIR = Path("/kaggle/working" if IS_KAGGLE else ".").resolve()
CHECKPOINT_ROOT = WORKING_DIR / "checkpoints"
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

print(f"Running on device: {DEVICE}")
print(f"Kaggle environment detected: {IS_KAGGLE}")
print(f"Checkpoint directory: {CHECKPOINT_ROOT}")
print(json.dumps(asdict(CFG), indent=2))

```

## Cell 4 (markdown)
## 1. Dataset Setup and Visualization

The notebook first looks for **Cityscapes** using the required Kaggle directory layout:

```python
DATASET_DIR = "/kaggle/input/cityscapes"
TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, "train/img")
TRAIN_LABELS_DIR = os.path.join(DATASET_DIR, "train/label")
VAL_IMAGES_DIR = os.path.join(DATASET_DIR, "val/img")
VAL_LABELS_DIR = os.path.join(DATASET_DIR, "val/label")
NUMBER_CLASSES = 21
```

If Cityscapes is not present, a **synthetic urban-scene fallback dataset** is generated so that the notebook still runs end-to-end without modification.

All images and masks are prepared for a final training size of **512 x 512**.


## Cell 5 (code)
```python
CLASS_NAMES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "background",
    "other",
]

CLASS_COLORS = np.array(
    [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
        (30, 30, 30),
        (255, 255, 255),
    ],
    dtype=np.uint8,
)

CITYSCAPES_RAW_TO_TRAIN = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def canonical_stem(path_like: str) -> str:
    stem = Path(path_like).stem
    suffixes = [
        "_leftImg8bit",
        "_gtFine_labelTrainIds",
        "_gtFine_labelIds",
        "_gtFine_color",
        "_labelTrainIds",
        "_labelIds",
        "_mask",
    ]
    for suffix in suffixes:
        stem = stem.replace(suffix, "")
    return stem

def prepare_mask(mask, num_classes: int = CFG.num_classes, ignore_index: int = CFG.ignore_index):
    mask = np.array(mask, dtype=np.int64)
    if mask.ndim == 3:
        mask = mask[..., 0]

    unique_values = np.unique(mask)
    max_value = int(unique_values.max()) if unique_values.size else 0
    valid_set = set(range(num_classes)) | {ignore_index}

    if set(unique_values.tolist()).issubset(valid_set):
        return mask

    raw_cityscapes_values = set(CITYSCAPES_RAW_TO_TRAIN.keys()) | {-1, ignore_index}
    if set(unique_values.tolist()).issubset(raw_cityscapes_values):
        remapped = np.full_like(mask, fill_value=ignore_index)
        for raw_id, train_id in CITYSCAPES_RAW_TO_TRAIN.items():
            remapped[mask == raw_id] = train_id
        remapped[mask == -1] = ignore_index
        remapped[mask == ignore_index] = ignore_index
        return remapped

    if max_value <= num_classes - 1:
        return mask

    clipped = mask.copy()
    clipped[(clipped < 0) | (clipped >= num_classes)] = ignore_index
    return clipped

def decode_segmentation_mask(mask):
    mask = np.array(mask)
    safe_mask = np.where((mask >= 0) & (mask < len(CLASS_COLORS)), mask, len(CLASS_COLORS) - 1)
    return CLASS_COLORS[safe_mask]

def denormalize_image(tensor):
    image = tensor.detach().cpu().float().clone()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = image * std + mean
    image = image.clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()

def overlay_segmentation(image, mask, alpha: float = 0.45):
    image = np.asarray(image, dtype=np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    color_mask = decode_segmentation_mask(mask).astype(np.float32) / 255.0
    return np.clip((1 - alpha) * image + alpha * color_mask, 0.0, 1.0)

def pick_indices(length: int, num_samples: int = 4):
    if length == 0:
        return []
    if length <= num_samples:
        return list(range(length))
    return np.linspace(0, length - 1, num_samples, dtype=int).tolist()

```

## Cell 6 (code)
```python
def build_lookup(root: Path, include_parent: bool = True):
    lookup = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
            parent_key = path.parent.relative_to(root).as_posix() if include_parent else ""
            key = "/".join(part for part in [parent_key, canonical_stem(path.name)] if part)
            lookup[key] = path
    return lookup

def collect_image_mask_pairs(images_dir: str, labels_dir: str):
    images_root = Path(images_dir)
    labels_root = Path(labels_dir)
    if not images_root.exists() or not labels_root.exists():
        return []

    image_lookup = build_lookup(images_root, include_parent=True)
    label_lookup = build_lookup(labels_root, include_parent=True)
    shared_keys = sorted(set(image_lookup) & set(label_lookup))

    if not shared_keys:
        image_lookup = build_lookup(images_root, include_parent=False)
        label_lookup = build_lookup(labels_root, include_parent=False)
        shared_keys = sorted(set(image_lookup) & set(label_lookup))

    return [(image_lookup[key], label_lookup[key]) for key in shared_keys]

class SegmentationTransform:
    def __init__(self, image_size: int = 512, train: bool = True):
        self.image_size = image_size
        self.train = train

    def _resize_and_crop(self, image, mask):
        if self.train:
            scaled_size = int(self.image_size * random.uniform(0.75, 1.25))
            scaled_size = max(scaled_size, self.image_size)
            image = image.resize((scaled_size, scaled_size), Image.BILINEAR)
            mask = mask.resize((scaled_size, scaled_size), Image.NEAREST)

            top = 0 if scaled_size == self.image_size else random.randint(0, scaled_size - self.image_size)
            left = 0 if scaled_size == self.image_size else random.randint(0, scaled_size - self.image_size)
            image = TF.crop(image, top, left, self.image_size, self.image_size)
            mask = TF.crop(mask, top, left, self.image_size, self.image_size)
        else:
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        return image, mask

    def __call__(self, image, mask):
        if self.train and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image, mask = self._resize_and_crop(image, mask)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = prepare_mask(mask)

        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        mask_tensor = torch.from_numpy(mask_np.astype(np.int64))
        return image_tensor, mask_tensor

class CityscapesSubsetDataset(Dataset):
    def __init__(self, pairs, transform=None, dataset_name: str = "Cityscapes"):
        self.pairs = list(pairs)
        self.transform = transform
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.pairs)

    def get_raw_sample(self, index):
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        meta = {"image_path": str(image_path), "mask_path": str(mask_path)}
        return image, mask, meta

    def __getitem__(self, index):
        image, mask, _ = self.get_raw_sample(index)
        if self.transform is None:
            image = image.resize((CFG.image_size, CFG.image_size), Image.BILINEAR)
            mask = mask.resize((CFG.image_size, CFG.image_size), Image.NEAREST)
            image_np = np.asarray(image, dtype=np.float32) / 255.0
            mask_np = prepare_mask(mask)
            image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
            mask_tensor = torch.from_numpy(mask_np.astype(np.int64))
            return image_tensor, mask_tensor
        return self.transform(image, mask)

class SyntheticUrbanDataset(Dataset):
    def __init__(self, length: int, transform=None, image_size: int = 768, split: str = "train", seed: int = 42):
        self.length = length
        self.transform = transform
        self.canvas_size = image_size
        self.split = split
        self.seed = seed + (0 if split == "train" else 10_000)

    def __len__(self):
        return self.length

    def _draw_vehicle(self, image_draw, mask_draw, rng, y_base, class_id):
        canvas = self.canvas_size
        car_w = rng.randint(int(canvas * 0.08), int(canvas * 0.18))
        car_h = rng.randint(int(canvas * 0.05), int(canvas * 0.09))
        x0 = rng.randint(0, canvas - car_w - 1)
        y0 = y_base - car_h
        color = tuple(int(c) for c in CLASS_COLORS[class_id])
        image_draw.rounded_rectangle([x0, y0, x0 + car_w, y0 + car_h], radius=8, fill=color)
        image_draw.rectangle([x0 + 8, y0 - 10, x0 + int(car_w * 0.7), y0 + 5], fill=(180, 220, 255))
        mask_draw.rounded_rectangle([x0, y0, x0 + car_w, y0 + car_h], radius=8, fill=int(class_id))

    def _draw_person(self, image_draw, mask_draw, rng, y_base):
        canvas = self.canvas_size
        x = rng.randint(int(canvas * 0.2), int(canvas * 0.8))
        body_h = rng.randint(int(canvas * 0.06), int(canvas * 0.1))
        color = tuple(int(c) for c in CLASS_COLORS[11])
        image_draw.ellipse([x - 8, y_base - body_h - 16, x + 8, y_base - body_h], fill=(245, 206, 170))
        image_draw.rectangle([x - 6, y_base - body_h, x + 6, y_base], fill=color)
        mask_draw.ellipse([x - 8, y_base - body_h - 16, x + 8, y_base - body_h], fill=11)
        mask_draw.rectangle([x - 6, y_base - body_h, x + 6, y_base], fill=11)

    def get_raw_sample(self, index):
        rng = random.Random(self.seed + index)
        canvas = self.canvas_size
        horizon = int(canvas * rng.uniform(0.42, 0.56))
        road_top = int(canvas * rng.uniform(0.56, 0.68))

        image = Image.new("RGB", (canvas, canvas), tuple(int(c) for c in CLASS_COLORS[10]))
        mask = Image.new("L", (canvas, canvas), 10)

        image_draw = ImageDraw.Draw(image)
        mask_draw = ImageDraw.Draw(mask)

        road_polygon = [(0, canvas), (canvas, canvas), (int(canvas * 0.64), road_top), (int(canvas * 0.36), road_top)]
        sidewalk_left = [(0, canvas), (int(canvas * 0.12), canvas), (int(canvas * 0.36), road_top), (0, road_top + 20)]
        sidewalk_right = [(canvas, canvas), (int(canvas * 0.88), canvas), (int(canvas * 0.64), road_top), (canvas, road_top + 20)]

        image_draw.polygon(road_polygon, fill=tuple(int(c) for c in CLASS_COLORS[0]))
        mask_draw.polygon(road_polygon, fill=0)
        image_draw.polygon(sidewalk_left, fill=tuple(int(c) for c in CLASS_COLORS[1]))
        image_draw.polygon(sidewalk_right, fill=tuple(int(c) for c in CLASS_COLORS[1]))
        mask_draw.polygon(sidewalk_left, fill=1)
        mask_draw.polygon(sidewalk_right, fill=1)

        for _ in range(rng.randint(4, 8)):
            x0 = rng.randint(0, int(canvas * 0.25))
            width = rng.randint(int(canvas * 0.08), int(canvas * 0.2))
            y0 = rng.randint(int(canvas * 0.18), horizon)
            color = tuple(int(c) for c in CLASS_COLORS[2] + rng.randint(-10, 10))
            color = tuple(max(0, min(255, val)) for val in color)
            image_draw.rectangle([x0, y0, x0 + width, canvas], fill=color)
            mask_draw.rectangle([x0, y0, x0 + width, canvas], fill=2)

        for _ in range(rng.randint(4, 8)):
            width = rng.randint(int(canvas * 0.08), int(canvas * 0.2))
            x0 = rng.randint(int(canvas * 0.75), canvas - width)
            y0 = rng.randint(int(canvas * 0.18), horizon)
            color = tuple(int(c) for c in CLASS_COLORS[2] + rng.randint(-10, 10))
            color = tuple(max(0, min(255, val)) for val in color)
            image_draw.rectangle([x0, y0, x0 + width, canvas], fill=color)
            mask_draw.rectangle([x0, y0, x0 + width, canvas], fill=2)

        for _ in range(rng.randint(2, 5)):
            x = rng.randint(0, canvas)
            r = rng.randint(int(canvas * 0.04), int(canvas * 0.08))
            image_draw.ellipse([x - r, horizon - r, x + r, horizon + r], fill=tuple(int(c) for c in CLASS_COLORS[8]))
            mask_draw.ellipse([x - r, horizon - r, x + r, horizon + r], fill=8)

        lane_y = [road_top + int((canvas - road_top) * frac) for frac in (0.15, 0.35, 0.6)]
        for y in lane_y:
            image_draw.line([(canvas // 2, y), (canvas // 2, y + 40)], fill=(255, 255, 180), width=6)
            mask_draw.line([(canvas // 2, y), (canvas // 2, y + 40)], fill=20, width=6)

        for _ in range(rng.randint(3, 6)):
            class_id = rng.choice([13, 13, 13, 14, 15])
            self._draw_vehicle(image_draw, mask_draw, rng, y_base=int(canvas * rng.uniform(0.72, 0.9)), class_id=class_id)

        for _ in range(rng.randint(1, 4)):
            self._draw_person(image_draw, mask_draw, rng, y_base=int(canvas * rng.uniform(0.65, 0.86)))

        for _ in range(rng.randint(1, 3)):
            x = rng.randint(int(canvas * 0.15), int(canvas * 0.85))
            pole_top = rng.randint(int(canvas * 0.15), int(canvas * 0.45))
            pole_bottom = road_top + rng.randint(0, 20)
            image_draw.line([(x, pole_top), (x, pole_bottom)], fill=tuple(int(c) for c in CLASS_COLORS[5]), width=5)
            image_draw.rectangle([x + 5, pole_top + 10, x + 28, pole_top + 35], fill=tuple(int(c) for c in CLASS_COLORS[7]))
            mask_draw.line([(x, pole_top), (x, pole_bottom)], fill=5, width=5)
            mask_draw.rectangle([x + 5, pole_top + 10, x + 28, pole_top + 35], fill=7)

        meta = {"split": self.split, "synthetic_index": index}
        return image, mask, meta

    def __getitem__(self, index):
        image, mask, _ = self.get_raw_sample(index)
        if self.transform is None:
            image = image.resize((CFG.image_size, CFG.image_size), Image.BILINEAR)
            mask = mask.resize((CFG.image_size, CFG.image_size), Image.NEAREST)
            image_np = np.asarray(image, dtype=np.float32) / 255.0
            mask_np = prepare_mask(mask)
            image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
            mask_tensor = torch.from_numpy(mask_np.astype(np.int64))
            return image_tensor, mask_tensor
        return self.transform(image, mask)

def subset_items(items, max_items: int, seed: int):
    items = list(items)
    if len(items) <= max_items:
        return items
    rng = random.Random(seed)
    selected = sorted(rng.sample(range(len(items)), max_items))
    return [items[idx] for idx in selected]

def prepare_datasets(config: Config):
    train_pairs = collect_image_mask_pairs(config.train_images_dir, config.train_labels_dir)
    val_pairs = collect_image_mask_pairs(config.val_images_dir, config.val_labels_dir)

    train_transform = SegmentationTransform(image_size=config.image_size, train=True)
    val_transform = SegmentationTransform(image_size=config.image_size, train=False)

    if train_pairs and val_pairs:
        train_pairs = subset_items(train_pairs, config.max_train_samples, config.seed)
        val_pairs = subset_items(val_pairs, config.max_val_samples, config.seed + 1)
        raw_train_dataset = CityscapesSubsetDataset(train_pairs, transform=None, dataset_name="Cityscapes subset")
        raw_val_dataset = CityscapesSubsetDataset(val_pairs, transform=None, dataset_name="Cityscapes subset")
        train_dataset = CityscapesSubsetDataset(train_pairs, transform=train_transform, dataset_name="Cityscapes subset")
        val_dataset = CityscapesSubsetDataset(val_pairs, transform=val_transform, dataset_name="Cityscapes subset")
        dataset_source = "Cityscapes subset"
    else:
        train_len = max(config.max_train_samples, 300)
        val_len = max(config.max_val_samples, 80)
        raw_train_dataset = SyntheticUrbanDataset(train_len, transform=None, image_size=768, split="train", seed=config.seed)
        raw_val_dataset = SyntheticUrbanDataset(val_len, transform=None, image_size=768, split="val", seed=config.seed)
        train_dataset = SyntheticUrbanDataset(train_len, transform=train_transform, image_size=768, split="train", seed=config.seed)
        val_dataset = SyntheticUrbanDataset(val_len, transform=val_transform, image_size=768, split="val", seed=config.seed)
        dataset_source = "Synthetic urban-scene fallback"

    dataset_info = {
        "source": dataset_source,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "image_size": config.image_size,
        "num_classes": config.num_classes,
    }
    return train_dataset, val_dataset, raw_train_dataset, raw_val_dataset, dataset_info

def build_dataloaders(train_dataset, val_dataset, batch_size: int):
    common_kwargs = {
        "num_workers": CFG.num_workers,
        "pin_memory": AMP_ENABLED,
    }
    if CFG.num_workers > 0:
        common_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, **common_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **common_kwargs)
    return train_loader, val_loader

def show_raw_dataset_grid(raw_dataset, split_name: str, num_samples: int = 4):
    indices = pick_indices(len(raw_dataset), num_samples)
    fig, axes = plt.subplots(len(indices), 2, figsize=(14, 4 * max(1, len(indices))))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, idx in enumerate(indices):
        image, mask, meta = raw_dataset.get_raw_sample(idx)
        mask_np = prepare_mask(mask)
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{split_name} Image #{idx}")
        axes[row, 1].imshow(decode_segmentation_mask(mask_np))
        axes[row, 1].set_title(f"{split_name} Mask #{idx}")
        for ax in axes[row]:
            ax.axis("off")
    plt.suptitle(f"{split_name}: Raw Samples", y=1.02, fontsize=18)
    plt.tight_layout()
    plt.show()

def show_preprocessing_comparison(raw_dataset, transformed_dataset, split_name: str, num_samples: int = 3):
    indices = pick_indices(len(raw_dataset), num_samples)
    fig, axes = plt.subplots(len(indices), 4, figsize=(22, 4 * max(1, len(indices))))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, idx in enumerate(indices):
        raw_image, raw_mask, _ = raw_dataset.get_raw_sample(idx)
        image_tensor, mask_tensor = transformed_dataset[idx]

        axes[row, 0].imshow(raw_image)
        axes[row, 0].set_title("Raw Image")
        axes[row, 1].imshow(decode_segmentation_mask(prepare_mask(raw_mask)))
        axes[row, 1].set_title("Raw Mask")
        axes[row, 2].imshow(denormalize_image(image_tensor))
        axes[row, 2].set_title("Augmented + Normalized\n(inverse-normalized for display)")
        axes[row, 3].imshow(decode_segmentation_mask(mask_tensor.numpy()))
        axes[row, 3].set_title("Processed Mask")
        for ax in axes[row]:
            ax.axis("off")

    plt.suptitle(f"{split_name}: Before vs After Preprocessing", y=1.02, fontsize=18)
    plt.tight_layout()
    plt.show()

```

## Cell 7 (markdown)
## 2. Preprocessing and Augmentation

The preprocessing pipeline applies:

- **ImageNet normalization** using mean and standard deviation
- **Random horizontal flip**
- **Random resized crop behaviour** through scaling followed by random cropping
- deterministic validation resizing to **512 x 512**

The visualizations below show:

- raw images and masks
- multiple samples in grid format
- before-vs-after preprocessing comparisons
- a batch sanity check from the PyTorch `DataLoader`


## Cell 8 (code)
```python
train_dataset, val_dataset, raw_train_dataset, raw_val_dataset, DATASET_INFO = prepare_datasets(CFG)

print("Dataset summary:")
print(json.dumps(DATASET_INFO, indent=2))

show_raw_dataset_grid(raw_train_dataset, split_name="Training", num_samples=4)
show_preprocessing_comparison(raw_train_dataset, train_dataset, split_name="Training", num_samples=3)

preview_loader, _ = build_dataloaders(train_dataset, val_dataset, batch_size=min(2, CFG.batch_size))
preview_images, preview_masks = next(iter(preview_loader))
print(f"Preview batch image tensor shape: {tuple(preview_images.shape)}")
print(f"Preview batch mask tensor shape: {tuple(preview_masks.shape)}")
print(f"Normalized batch mean: {preview_images.mean().item():.4f}")
print(f"Normalized batch std: {preview_images.std().item():.4f}")

```

## Cell 9 (markdown)
## 3. Model Architectures

Two segmentation models are implemented and compared.

### (A) CNN Baseline
- **U-Net** decoder
- **ResNet34** encoder by default (changeable to ResNet50)
- pretrained encoder weights are used when available

### (B) Transformer Model
- **SegFormer-B2**
- pretrained Hugging Face weights are used when available

Both models:

- accept the same input size
- produce multi-class segmentation logits
- are summarized below with parameter counts and output shapes


## Cell 10 (code)
```python
RESNET_BUILDERS = {
    "resnet34": (models.resnet34, ResNet34_Weights.DEFAULT),
    "resnet50": (models.resnet50, ResNet50_Weights.DEFAULT),
}

RESNET_CHANNELS = {
    "resnet34": [64, 64, 128, 256, 512],
    "resnet50": [64, 256, 512, 1024, 2048],
}

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBNReLU(out_channels + skip_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

def load_resnet_backbone(encoder_name: str, pretrained: bool = True):
    builder, weights = RESNET_BUILDERS[encoder_name]
    if not pretrained:
        return builder(weights=None), False
    try:
        backbone = builder(weights=weights)
        return backbone, True
    except Exception as exc:
        print(f"Could not load pretrained {encoder_name} weights: {exc}")
        print("Falling back to randomly initialized ResNet encoder.")
        return builder(weights=None), False

class ResNetUNet(nn.Module):
    def __init__(self, encoder_name: str = "resnet34", num_classes: int = 21, pretrained: bool = True):
        super().__init__()
        backbone, pretrained_loaded = load_resnet_backbone(encoder_name, pretrained=pretrained)
        channels = RESNET_CHANNELS[encoder_name]
        decoder_channels = [256, 128, 64, 64] if encoder_name == "resnet34" else [512, 256, 128, 64]

        self.pretrained_loaded = pretrained_loaded
        self.encoder_name = encoder_name

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.decoder4 = DecoderBlock(channels[4], channels[3], decoder_channels[0])
        self.decoder3 = DecoderBlock(decoder_channels[0], channels[2], decoder_channels[1])
        self.decoder2 = DecoderBlock(decoder_channels[1], channels[1], decoder_channels[2])
        self.decoder1 = DecoderBlock(decoder_channels[2], channels[0], decoder_channels[3])

        self.segmentation_head = nn.Sequential(
            ConvBNReLU(decoder_channels[3], decoder_channels[3]),
            nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1),
        )

        self.gradcam_target_layer = self.encoder4[-1]

    def forward(self, x):
        input_size = x.shape[-2:]

        x0 = self.stem(x)
        x1 = self.encoder1(self.pool(x0))
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        d4 = self.decoder4(x4, x3)
        d3 = self.decoder3(d4, x2)
        d2 = self.decoder2(d3, x1)
        d1 = self.decoder1(d2, x0)

        logits = self.segmentation_head(d1)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits

class SegFormerWrapper(nn.Module):
    def __init__(self, num_classes: int = 21, pretrained_model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512"):
        super().__init__()
        id2label = {idx: label for idx, label in enumerate(CLASS_NAMES)}
        label2id = {label: idx for idx, label in id2label.items()}

        self.pretrained_name = pretrained_model_name
        self.pretrained_loaded = False
        self.model = None
        last_exception = None

        for local_only in (True, False):
            try:
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    pretrained_model_name,
                    num_labels=num_classes,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_mismatched_sizes=True,
                    local_files_only=local_only,
                )
                self.pretrained_loaded = True
                break
            except Exception as exc:
                last_exception = exc

        if self.model is None:
            print(f"Could not load pretrained SegFormer-B2 weights: {last_exception}")
            print("Falling back to randomly initialized SegFormer-B2 configuration.")
            config = SegformerConfig(num_labels=num_classes, id2label=id2label, label2id=label2id)
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x):
        outputs = self.model(pixel_values=x, return_dict=True)
        logits = outputs.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def forward_with_outputs(self, x, output_attentions: bool = False, output_hidden_states: bool = False):
        outputs = self.model(
            pixel_values=x,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = outputs.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return outputs, logits

def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable

def summarize_model(model, model_name: str, input_size=(1, 3, 512, 512)):
    total_params, trainable_params = count_parameters(model)
    print("=" * 100)
    print(model_name)
    print("=" * 100)
    print(model)
    dummy_input = torch.randn(*input_size, device=DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 100)

```

## Cell 11 (code)
```python
cnn_model = ResNetUNet(encoder_name=CFG.encoder_name, num_classes=CFG.num_classes, pretrained=True).to(DEVICE)
segformer_model = SegFormerWrapper(num_classes=CFG.num_classes).to(DEVICE)

print(f"U-Net encoder pretrained loaded: {cnn_model.pretrained_loaded}")
print(f"SegFormer pretrained loaded: {segformer_model.pretrained_loaded}")

summarize_model(cnn_model, model_name=f"U-Net with {CFG.encoder_name} encoder", input_size=(1, 3, CFG.image_size, CFG.image_size))
summarize_model(segformer_model, model_name="SegFormer-B2", input_size=(1, 3, CFG.image_size, CFG.image_size))

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

```

## Cell 12 (markdown)
## 4. Training Setup

The training pipeline below uses:

- **Loss:** `CrossEntropyLoss` for multi-class segmentation
- **Optimizer:** `AdamW`
- **Learning rate:** `1e-4`
- **Mixed precision:** `torch.cuda.amp`
- **Metrics:** mean **IoU** and mean **Dice score**
- **Progress bars:** `tqdm`
- **Checkpointing:** saved after **every epoch**
- **Resume support:** training can continue from the latest checkpoint automatically

After every epoch, the notebook:

- prints train and validation metrics
- plots loss and metric curves
- shows sample predictions and overlay visualizations


## Cell 13 (code)
```python
def sync_cuda():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

def update_confusion_matrix(confmat, preds, targets, num_classes: int, ignore_index: int):
    preds = preds.view(-1).to(torch.int64)
    targets = targets.view(-1).to(torch.int64)
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]
    if targets.numel() == 0:
        return confmat
    indices = targets * num_classes + preds
    confmat += torch.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return confmat

def compute_metrics_from_confmat(confmat):
    confmat = confmat.float()
    true_positive = torch.diag(confmat)
    false_positive = confmat.sum(dim=0) - true_positive
    false_negative = confmat.sum(dim=1) - true_positive

    union = true_positive + false_positive + false_negative
    dice_denom = 2 * true_positive + false_positive + false_negative

    iou = torch.where(union > 0, true_positive / union, torch.nan)
    dice = torch.where(dice_denom > 0, (2 * true_positive) / dice_denom, torch.nan)

    return {
        "iou": torch.nanmean(iou).item(),
        "dice": torch.nanmean(dice).item(),
    }

def latest_checkpoint_path(checkpoint_dir: Path):
    candidates = list(checkpoint_dir.glob("epoch_*.pth"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: int(path.stem.split("_")[-1]))

def move_optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

def save_checkpoint(model_name: str, epoch: int, model, optimizer, scaler, history, checkpoint_dir: Path):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1:02d}.pth"
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "history": history,
        "config": asdict(CFG),
        "model_name": model_name,
    }
    torch.save(payload, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_latest_checkpoint(model, optimizer, scaler, checkpoint_dir: Path):
    checkpoint_path = latest_checkpoint_path(checkpoint_dir)
    if checkpoint_path is None:
        return 0, defaultdict(list), None

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    move_optimizer_to_device(optimizer, DEVICE)
    if "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])

    history = defaultdict(list)
    for key, values in checkpoint.get("history", {}).items():
        history[key] = list(values)

    start_epoch = int(checkpoint["epoch"]) + 1
    return start_epoch, history, checkpoint_path

def plot_training_history(history, model_name: str):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    if len(epochs) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    axes[0].plot(epochs, history["train_loss"], marker="o", linewidth=2.5, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], marker="s", linewidth=2.5, label="Val Loss")
    axes[0].set_title(f"{model_name}: Training vs Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_iou"], marker="o", linewidth=2.5, label="Train IoU")
    axes[1].plot(epochs, history["val_iou"], marker="s", linewidth=2.5, label="Val IoU")
    axes[1].plot(epochs, history["train_dice"], marker="^", linewidth=2.5, label="Train Dice")
    axes[1].plot(epochs, history["val_dice"], marker="d", linewidth=2.5, label="Val Dice")
    axes[1].set_title(f"{model_name}: IoU and Dice Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

@torch.no_grad()
def show_sample_predictions(model, dataset, model_name: str, num_samples: int = 3):
    model.eval()
    indices = pick_indices(len(dataset), num_samples)
    fig, axes = plt.subplots(len(indices), 4, figsize=(22, 5 * max(1, len(indices))))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, idx in enumerate(indices):
        image_tensor, mask_tensor = dataset[idx]
        logits = model(image_tensor.unsqueeze(0).to(DEVICE))
        pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        image_np = denormalize_image(image_tensor)
        gt_mask = mask_tensor.cpu().numpy()
        overlay = overlay_segmentation(image_np, pred_mask)

        axes[row, 0].imshow(image_np)
        axes[row, 0].set_title("Input Image")
        axes[row, 1].imshow(decode_segmentation_mask(gt_mask))
        axes[row, 1].set_title("Ground Truth")
        axes[row, 2].imshow(decode_segmentation_mask(pred_mask))
        axes[row, 2].set_title("Prediction")
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title("Prediction Overlay")

        for ax in axes[row]:
            ax.axis("off")

    plt.suptitle(f"{model_name}: Validation Predictions", y=1.02, fontsize=18)
    plt.tight_layout()
    plt.show()

def run_epoch(model, loader, criterion, optimizer=None, scaler=None, training: bool = True):
    model.train(training)
    confmat = torch.zeros((CFG.num_classes, CFG.num_classes), dtype=torch.long)
    running_loss = 0.0
    samples_seen = 0

    sync_cuda()
    start_time = time.perf_counter()
    desc = "Train" if training else "Val"
    progress = tqdm(loader, desc=desc, leave=False)

    for images, masks in progress:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with autocast(enabled=AMP_ENABLED):
                logits = model(images)
                loss = criterion(logits, masks)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        running_loss += loss.item() * images.size(0)
        samples_seen += images.size(0)

        preds = logits.argmax(dim=1).detach().cpu()
        targets = masks.detach().cpu()
        confmat = update_confusion_matrix(confmat, preds, targets, CFG.num_classes, CFG.ignore_index)
        running_metrics = compute_metrics_from_confmat(confmat)
        avg_loss = running_loss / max(1, samples_seen)

        progress.set_postfix(
            loss=f"{avg_loss:.4f}",
            iou=f"{running_metrics['iou']:.4f}",
            dice=f"{running_metrics['dice']:.4f}",
        )

    sync_cuda()
    epoch_time = time.perf_counter() - start_time
    metrics = compute_metrics_from_confmat(confmat)
    metrics["loss"] = running_loss / max(1, samples_seen)
    metrics["time"] = epoch_time
    return metrics

def train_model(model, model_name: str, train_dataset, val_dataset, batch_size: int):
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss(ignore_index=CFG.ignore_index)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scaler = GradScaler(enabled=AMP_ENABLED)

    checkpoint_dir = CHECKPOINT_ROOT / model_name
    history = defaultdict(list)
    start_epoch = 0

    if CFG.resume_training:
        start_epoch, loaded_history, checkpoint_path = load_latest_checkpoint(model, optimizer, scaler, checkpoint_dir)
        history.update(loaded_history)
        if checkpoint_path is not None:
            print(f"Resuming {model_name} from checkpoint: {checkpoint_path}")

    if start_epoch >= CFG.epochs:
        print(f"{model_name} already has {start_epoch} epochs completed. Skipping retraining.")
        plot_training_history(history, model_name)
        show_sample_predictions(model, val_dataset, model_name=model_name, num_samples=3)
        return {"history": history, "checkpoint_dir": checkpoint_dir}

    total_start = time.perf_counter()

    for epoch in range(start_epoch, CFG.epochs):
        print(f"\nEpoch {epoch + 1}/{CFG.epochs} - {model_name}")
        train_metrics = run_epoch(model, train_loader, criterion, optimizer=optimizer, scaler=scaler, training=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer=None, scaler=None, training=False)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_iou"].append(train_metrics["iou"])
        history["val_iou"].append(val_metrics["iou"])
        history["train_dice"].append(train_metrics["dice"])
        history["val_dice"].append(val_metrics["dice"])
        history["train_epoch_time"].append(train_metrics["time"])
        history["val_epoch_time"].append(val_metrics["time"])
        history["epoch_wall_time"].append(train_metrics["time"] + val_metrics["time"])

        print(
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train IoU: {train_metrics['iou']:.4f} | "
            f"Train Dice: {train_metrics['dice']:.4f}"
        )
        print(
            f"Val Loss:   {val_metrics['loss']:.4f} | "
            f"Val IoU:   {val_metrics['iou']:.4f} | "
            f"Val Dice:   {val_metrics['dice']:.4f}"
        )
        print(f"Epoch time: {history['epoch_wall_time'][-1]:.2f} seconds")

        save_checkpoint(model_name, epoch, model, optimizer, scaler, history, checkpoint_dir)
        plot_training_history(history, model_name)
        show_sample_predictions(model, val_dataset, model_name=model_name, num_samples=3)

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    total_time = time.perf_counter() - total_start
    history["total_training_time"] = [total_time]
    history_path = WORKING_DIR / f"{model_name}_history.json"
    with open(history_path, "w", encoding="utf-8") as fp:
        json.dump({key: list(value) for key, value in history.items()}, fp, indent=2)
    print(f"Training history saved to: {history_path}")
    print(f"Total training time for {model_name}: {total_time / 60:.2f} minutes")

    return {"history": history, "checkpoint_dir": checkpoint_dir}

```

## Cell 14 (markdown)
## 5. Train CNN Baseline: U-Net with ResNet Encoder

The CNN baseline uses a U-Net decoder with a pretrained **ResNet34** encoder by default.  
Each epoch prints metrics, shows progress bars, saves a checkpoint, and visualizes updated results.


## Cell 15 (code)
```python
cnn_results = train_model(
    model=cnn_model,
    model_name=f"unet_{CFG.encoder_name}",
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=CFG.batch_size,
)

```

## Cell 16 (markdown)
## 6. Train Transformer Model: SegFormer-B2

The transformer baseline uses **SegFormer-B2** with pretrained weights when available.  
To keep memory usage safe on Kaggle T4, the default batch size is smaller than the CNN baseline.


## Cell 17 (code)
```python
segformer_results = train_model(
    model=segformer_model,
    model_name="segformer_b2",
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=CFG.segformer_batch_size,
)

```

## Cell 18 (markdown)
## 7. Explainable AI

This section adds interpretable visual evidence for both model families:

- **Grad-CAM** on the U-Net encoder to highlight important spatial regions
- **Attention map visualization** for SegFormer-B2 to inspect transformer focus patterns


## Cell 19 (code)
```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        pred_mask = logits.argmax(dim=1)

        if class_idx is None:
            values, counts = pred_mask[0].unique(return_counts=True)
            ranking = torch.argsort(counts, descending=True)
            class_idx = int(values[ranking[0]].item())

        target_score = logits[:, class_idx].mean()
        target_score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, pred_mask[0].detach().cpu().numpy(), class_idx

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

def show_gradcam_result(model, dataset, sample_index: int = 0):
    image_tensor, mask_tensor = dataset[sample_index]
    gradcam = GradCAM(model, model.gradcam_target_layer)
    cam, pred_mask, class_idx = gradcam.generate(image_tensor.unsqueeze(0).to(DEVICE))
    gradcam.close()

    image_np = denormalize_image(image_tensor)
    gt_mask = mask_tensor.numpy()

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(image_np)
    axes[0].set_title("Input Image")
    axes[1].imshow(decode_segmentation_mask(gt_mask))
    axes[1].set_title("Ground Truth")
    axes[2].imshow(image_np)
    axes[2].imshow(cam, cmap="jet", alpha=0.5)
    axes[2].set_title(f"Grad-CAM ({CLASS_NAMES[class_idx]})")
    axes[3].imshow(overlay_segmentation(image_np, pred_mask))
    axes[3].set_title("Prediction Overlay")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def show_segformer_attention(model, dataset, sample_index: int = 0):
    model.eval()
    image_tensor, mask_tensor = dataset[sample_index]
    outputs, logits = model.forward_with_outputs(
        image_tensor.unsqueeze(0).to(DEVICE),
        output_attentions=True,
        output_hidden_states=False,
    )
    pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    attention_tensors = outputs.attentions
    if attention_tensors is None or len(attention_tensors) == 0:
        raise RuntimeError("SegFormer attention tensors were not returned.")

    attention = attention_tensors[-1]
    attention_map = attention.mean(dim=1).mean(dim=1)[0]
    side = int(math.sqrt(attention_map.numel()))
    attention_map = attention_map[: side * side].reshape(side, side)
    attention_map = attention_map.unsqueeze(0).unsqueeze(0)
    attention_map = F.interpolate(
        attention_map,
        size=(CFG.image_size, CFG.image_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

    image_np = denormalize_image(image_tensor)
    gt_mask = mask_tensor.numpy()

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(image_np)
    axes[0].set_title("Input Image")
    axes[1].imshow(decode_segmentation_mask(gt_mask))
    axes[1].set_title("Ground Truth")
    axes[2].imshow(image_np)
    axes[2].imshow(attention_map, cmap="magma", alpha=0.55)
    axes[2].set_title("SegFormer Attention Map")
    axes[3].imshow(overlay_segmentation(image_np, pred_mask))
    axes[3].set_title("Prediction Overlay")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

print("Grad-CAM for the CNN model")
show_gradcam_result(cnn_model, val_dataset, sample_index=0)

print("Attention-map visualization for SegFormer-B2")
show_segformer_attention(segformer_model, val_dataset, sample_index=0)

```

## Cell 20 (markdown)
## 8. Time Analysis and Final Model Comparison

This section measures and compares:

- training time per epoch
- total training time
- inference time per image
- final validation IoU and Dice scores

Results are presented in both **tables** and **plots** for a presentation-ready summary.


## Cell 21 (code)
```python
def benchmark_inference_time(model, dataset, runs: int = 20):
    model.eval()
    sample_indices = pick_indices(len(dataset), min(runs, len(dataset)))
    prepared_inputs = [dataset[idx][0].unsqueeze(0).to(DEVICE) for idx in sample_indices]

    with torch.no_grad():
        for tensor in prepared_inputs[: min(5, len(prepared_inputs))]:
            _ = model(tensor)

        sync_cuda()
        start = time.perf_counter()
        for tensor in prepared_inputs:
            _ = model(tensor)
        sync_cuda()
        total_time = time.perf_counter() - start

    return total_time / max(1, len(prepared_inputs))

def summarize_history(model_label: str, history, inference_seconds: float):
    best_epoch = int(np.argmax(history["val_iou"])) + 1
    return {
        "Model": model_label,
        "Best Epoch": best_epoch,
        "Best Val IoU": float(np.max(history["val_iou"])),
        "Best Val Dice": float(np.max(history["val_dice"])),
        "Avg Epoch Time (s)": float(np.mean(history["epoch_wall_time"])),
        "Total Training Time (min)": float(np.sum(history["epoch_wall_time"]) / 60.0),
        "Inference Time / Image (ms)": float(inference_seconds * 1000.0),
    }

cnn_inference_time = benchmark_inference_time(cnn_model, val_dataset, runs=20)
segformer_inference_time = benchmark_inference_time(segformer_model, val_dataset, runs=20)

cnn_history = cnn_results["history"]
segformer_history = segformer_results["history"]

comparison_df = pd.DataFrame(
    [
        summarize_history(f"U-Net ({CFG.encoder_name})", cnn_history, cnn_inference_time),
        summarize_history("SegFormer-B2", segformer_history, segformer_inference_time),
    ]
)

comparison_path = WORKING_DIR / "model_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)

print(f"Comparison table saved to: {comparison_path}")
display(
    comparison_df.style.format(
        {
            "Best Val IoU": "{:.4f}",
            "Best Val Dice": "{:.4f}",
            "Avg Epoch Time (s)": "{:.2f}",
            "Total Training Time (min)": "{:.2f}",
            "Inference Time / Image (ms)": "{:.2f}",
        }
    )
)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
metrics_to_plot = [
    ("Best Val IoU", "Validation IoU", axes[0, 0]),
    ("Best Val Dice", "Validation Dice", axes[0, 1]),
    ("Total Training Time (min)", "Total Training Time (minutes)", axes[1, 0]),
    ("Inference Time / Image (ms)", "Inference Time per Image (ms)", axes[1, 1]),
]

for column, title, ax in metrics_to_plot:
    sns.barplot(data=comparison_df, x="Model", y=column, palette="viridis", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=10)

plt.tight_layout()
plt.show()

max_epochs = max(len(cnn_history["epoch_wall_time"]), len(segformer_history["epoch_wall_time"]))
epoch_axis = np.arange(1, max_epochs + 1)
timing_df = pd.DataFrame(
    {
        "Epoch": epoch_axis,
        f"U-Net ({CFG.encoder_name})": pd.Series(cnn_history["epoch_wall_time"]),
        "SegFormer-B2": pd.Series(segformer_history["epoch_wall_time"]),
    }
)

plt.figure(figsize=(14, 6))
plt.plot(timing_df["Epoch"], timing_df[f"U-Net ({CFG.encoder_name})"], marker="o", linewidth=2.5, label=f"U-Net ({CFG.encoder_name})")
plt.plot(timing_df["Epoch"], timing_df["SegFormer-B2"], marker="s", linewidth=2.5, label="SegFormer-B2")
plt.title("Training Time per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Seconds")
plt.legend()
plt.tight_layout()
plt.show()

print("Project summary:")
print("- Cityscapes subset used when available; otherwise synthetic urban scenes kept the notebook runnable.")
print("- Both models used the same 512x512 input size and identical core metrics (IoU and Dice).")
print("- Checkpoints were written after each epoch and histories were saved to /kaggle/working.")
print("- Increase CFG.epochs to 30-50 on stronger hardware for a deeper comparative study.")

```

## Cell 22 (markdown)
## 9. Conclusion

This notebook provides a full semantic segmentation project pipeline suitable for a deep learning course submission.

It demonstrates:

- efficient data loading and preprocessing
- comparative training of a **CNN** and a **Vision Transformer**
- checkpointing and resume logic
- prediction visualization and explainability
- runtime and accuracy comparison

For a final report, you can discuss:

- whether the CNN or transformer converged faster
- which model produced cleaner boundaries and object separation
- the trade-off between segmentation quality and computational cost


