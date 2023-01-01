# Dataset
# -images/
#   - 0000 - 000000000.png
#           - 000000001.png
import random
import json
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import numpy as np

from ..model.converter import LabelConverter


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
        self.reading_to_signs = {}
        self.sign_to_index = {}
        self.transform = transform
        self._converter = LabelConverter(label_max_length, target_signs_file_path)

        self._load_target_signs(target_signs_file_path)

    def _load_target_signs(self, target_signs_file_path: str):
        with open(target_signs_file_path) as f:
            loaded = json.load(f)

        for signs in sorted(loaded):
            sign_indices = []  # list of int sign indices
            for sign in signs.split("."):
                if sign not in self.sign_to_index:
                    idx: int = len(self.sign_to_index) + 1  # 0 is for blank
                    self.sign_to_index[sign] = idx
                sign_indices.append(idx)

            for reading in loaded[signs]["readings"]:
                self.reading_to_signs[reading["reading"]] = sign_indices

        self.space_index = len(self.sign_to_index)
        self.unk_index = len(self.sign_to_index) + 1

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
        # image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
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
            (0, stacked.height // 2 - 48, stacked.width, stacked.height // 2 + 48)
        )

        if self.transform:
            image = self.transform(image)

        # Text
        text_path = (
            Path(self.texts_root_dir) / f"{index//(10**3):04d}" / f"{index:09d}.json"
        )

        with open(text_path) as f:
            loaded = json.load(f)

        target = self._converter.encode(loaded)
        target = torch.LongTensor(target)

        return image, target
