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
        img_height: int = 96,
        img_width: int = 64 * 21,
        transform=None,
    ):
        self.images_root_dir = images_root_dir
        self._label_max_len = label_max_length
        self.img_height = img_height
        self.img_width = img_width
        self._PAD_TOKEN = "[PAD]"
        self._GO_TOKEN = "[GO]"
        self._STOP_TOKEN = "[STOP]"
        self.reading_to_signs = {}
        self.sign_to_index = {
            self._PAD_TOKEN: 0,
            self._GO_TOKEN: 1,
            self._STOP_TOKEN: 2,
        }

        self.unk_index = -1
        self.space_index = -2
        self.transform = transform

        self._load_target_signs(target_signs_file_path)

        text_raw_data = [
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
        self.target_labels = self._generate_labels(text_raw_data)

    def _load_target_signs(self, target_signs_file_path: str):
        with open(target_signs_file_path) as f:
            loaded = json.load(f)

        for signs in sorted(loaded):
            sign_indices = []  # list of int sign indices
            for sign in signs.split("."):
                if sign not in self.sign_to_index:
                    self.sign_to_index[sign] = len(self.sign_to_index)
                sign_indices.append(self.sign_to_index[sign])

            for reading in loaded[signs]["readings"]:
                self.reading_to_signs[reading["reading"]] = sign_indices

        self.space_index = len(self.sign_to_index) + 1
        self.unk_index = len(self.sign_to_index) + 2

    def __len__(self):
        # return len(self.target_labels)
        return len(self.target_labels)

    def _get_image_path(self, index):
        image_path = Path(self.images_root_dir) / f"valid_{index + 1:03d}.png"
        return image_path

    def __getitem__(self, index):
        # Image
        image_path = self._get_image_path(index)

        image = Image.open(str(image_path)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Text
        target = self.target_labels[index]

        return image, target

    def _generate_labels(self, text_raw_data):
        targets = []
        for line in text_raw_data:
            target = [self.sign_to_index[self._GO_TOKEN]]  # 1st index is for GO TOKEN
            for reading in line:
                if reading == " ":
                    target.append(self.space_index)
                elif reading in self.reading_to_signs:
                    target.extend(self.reading_to_signs[reading])
                else:
                    target.append(self.unk_index)

            target = target[:-1]  # remove last space
            target.append(self.sign_to_index[self._STOP_TOKEN])  # end with STOP TOKEN
            text = torch.tensor(target)
            text_with_pad = torch.zeros(self._label_max_len + 1, dtype=torch.long)
            text_with_pad[: len(text)] = text

            targets.append(text_with_pad)

        return targets
