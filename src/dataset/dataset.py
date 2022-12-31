# Dataset
# -images/
#   - 0000 - 000000000.png
#           - 000000001.png
import random
import json
from pathlib import Path
from typing import List

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
        self.unk_index = -1
        self.space_index = -2
        self.transform = transform
        self._converter = LabelConverter(label_max_length)

        self._load_target_signs(target_signs_file_path)

    def _load_target_signs(self, target_signs_file_path: str):
        with open(target_signs_file_path) as f:
            loaded = json.load(f)

        for signs in sorted(loaded):
            sign_indices = []  # list of int sign indices
            for sign in signs.split("."):
                if sign not in self.sign_to_index:
                    self.sign_to_index[sign] = (
                        len(self.sign_to_index) + 1
                    )  # 0 is for blank
                sign_indices.append(self.sign_to_index[sign])

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
        target = self._load_text(text_path)
        target = self._converter.encode(target)

        target_length = [len(target)]
        target = torch.LongTensor(target)

        target_length = torch.LongTensor(target_length)

        return image, target, target_length

    def _load_text(self, path: Path) -> List[int]:
        """
        Read annotation json file

        Args:
            path (str): Annotaiton file path.

        Returns:
            The list of character index.
        """
        with open(path) as f:
            loaded = json.load(f)

        target = []
        for line in loaded["line"]:
            for words in line["signs"]:
                for reading_dict in words:
                    reading = reading_dict["reading"]
                    if reading in self.reading_to_signs:
                        target.extend(self.reading_to_signs[reading])
                    else:
                        target.append(self.unk_index)
                target.append(self.space_index)

        return target[:-1]  # remove last space


def synth_cuneiform_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths
