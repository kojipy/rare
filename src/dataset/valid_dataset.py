import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from ..model.converter import LabelConverter


class SyntheticCuneiformValidationLineImage(Dataset):
    def __init__(
        self,
        *,
        target_signs_file_path: str,
        images_root_dir: str,
        label_max_length: int,
        img_height: int = 64,
        img_width: int = 64 * 21,
        transform=None,
    ):
        self.images_root_dir = images_root_dir
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform

        self._converter = LabelConverter(label_max_length, target_signs_file_path)

        self._text_raw_data = [
            ["EGIR", "pa", " ", "ti", "an", "zi"],  # OK
            [
                "nu",
                "uš",
                "ša",
                "an",
                " ",
                "A",
                "NA",
                " ",
                "NINDA",
                "GUR",
                "RA",
                " ",
                "ŠE",
                "ER",
            ],  # OK
            [
                "pé",
                "ra",
                "an",
                " ",
                "kat",
                "ta",
                "ma",
                " ",
                "ki",
                "ne",
                " ",
                "i",
                "ia",
                "mi",
            ],  # OK
            ["NINDA", "šar", "li", "in", "na", " ", "te", "eḫ", "ḫi"],  # OK
            ["a", "da", "an", "zi", " ", "a", "ku", "wa", "an", "zi"],  # OK
            ["MAḪ", "aš", " ", "LUGAL", "i", " ", "MUNUS", "LUGAL", "i"],  # OK
            [
                "ap",
                "pa",
                "an",
                "zi",
                " ",
                "pa",
                "ri",
                "li",
                "ia",
                "aš",
                "ša",
                " ",
                "MUŠEN",
                "ḪI",
                "A",
            ],  # OK
            [
                "nu",
                "za",
                " ",
                "wa",
                "ar",
                "ap",
                "zi",
                " ",
                "nam",
                "ma",
                "za",
                "a",
                "pa",
                "a",
                "aš",
            ],  # OK
            [
                "9",
                "NA@4",
                "pa",
                "aš",
                "ši",
                "la",
                "aš",
                " ",
                "A",
                "ŠÀ",
                " ",
                "te",
                "ri",
                "ip",
                "pí",
                "aš",
            ],  # OK
        ]

    def __len__(self):
        # return len(self.target_labels)
        return len(self._text_raw_data)

    def _get_image_path(self, index):
        image_path = Path(self.images_root_dir) / f"valid_{index:03d}.png"
        return image_path

    def __getitem__(self, index):
        # Image
        image_path = self._get_image_path(index=index)

        image = Image.open(str(image_path)).convert("RGB")
        # image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)

        if self.transform:
            image = self.transform(image)

        text = self._text_raw_data[index]
        target = self._converter(text)

        return image, target
