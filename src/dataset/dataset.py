# Dataset
# -images/
#   - 0000 - 000000000.png
#           - 000000001.png
import random
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from ..model.converter import LabelConverter
from .label_file import LabelFile


def my_resize(v, minv, maxv):
    scale = 1 + random.uniform(-0.2, 0.2)
    return int(max(minv, min(maxv, scale * v)))


def vstack(images):
    if len(images) == 0:
        raise ValueError("Need 0 or more images")

    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(img) for img in images]
    width = max([img.size[0] for img in images])
    height = sum([img.size[1] for img in images])
    stacked = Image.new(images[0].mode, (width, height))

    y_pos = 0
    for img in images:
        stacked.paste(img, (0, y_pos))
        y_pos += img.size[1]
    return stacked


class SyntheticCuneiformLineImage(Dataset):
    def __init__(
        self,
        *,
        target_signs_file_path: str,
        images_root_dir: str,
        texts_root_dir: str,
        first_idx: int,
        last_idx: int,
        label_max_length: int,
        img_height: int = 96,
        img_width: int = 64 * 24,
        transform=None,
    ):
        assert first_idx >= 0
        assert last_idx >= 0
        assert first_idx <= last_idx

        self.first_idx = first_idx
        self.last_idx = last_idx
        self.images_root_dir = images_root_dir
        self.texts_root_dir = texts_root_dir
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform
        self._label_max_length = label_max_length
        self._converter = LabelConverter(self._label_max_length, target_signs_file_path)

    def _get_image_path(self, index):
        image_path = (
            Path(self.images_root_dir) / f"{index//(10**3):04d}" / f"{index:09d}.png"
        )
        return image_path

    def __len__(self):
        return self.last_idx - self.first_idx + 1

    def __getitem__(self, index: int):
        # Image
        image_path = self._get_image_path(index)

        image = Image.open(str(image_path)).convert("RGB")
        width = int(image.width * (self.img_height / image.height))
        width = my_resize(width, 128, 1536)

        image = image.resize((width, self.img_height), resample=Image.BILINEAR)
        image = ImageOps.pad(
            image, (self.img_width, self.img_height), color=(0, 0, 0), centering=(0, 0)
        )

        img_above = Image.open(
            str(self._get_image_path(random.randint(self.first_idx, self.last_idx)))
        ).convert("RGB")
        img_below = Image.open(
            str(self._get_image_path(random.randint(self.first_idx, self.last_idx)))
        ).convert("RGB")

        stacked = vstack([img_above, image, img_below])
        image = stacked.crop(
            (
                0,
                stacked.height // 2 - int(image.height / 2 * 1.75),
                stacked.width,
                stacked.height // 2 + int(image.height / 2 * 1.75),
            )
        )

        if self.transform:
            image = self.transform(image)

        # Text
        text_path = (
            Path(self.texts_root_dir) / f"{index//(10**3):04d}" / f"{index:09d}.json"
        )

        lf = LabelFile(str(text_path))
        target = self.encode(lf.text)

        return image, target

    def decode(self, text: List[int]):
        return self._converter.decode(text)

    def encode(self, text: List[str]):
        return self._converter.encode(text)
