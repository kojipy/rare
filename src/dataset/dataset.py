import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from PIL import Image

# Dataset
# -images/
#   - 0000 - 000000000.png
#           - 000000001.png


class SyntheticCuneiformValidationLineImage(Dataset):
    def __init__(
        self,
        *,
        target_signs_file_path: str,
        images_root_dir: str,
        text_raw_data: List,
        img_height: int = 64,
        img_width: int = 64 * 21,
        transform=None,
    ):
        self.images_root_dir = images_root_dir
        self.img_height = img_height
        self.img_width = img_width
        self.reading_to_signs = {}
        self.sign_to_index = {}
        self.unk_index = -1
        self.space_index = -2
        self.transform = transform

        self._load_target_signs(target_signs_file_path)
        self.target_labels = self._generate_labels(text_raw_data)

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

    def __len__(self):
        return len(self.target_labels)

    def __getitem__(self, index):
        # Image
        image_path = Path(self.images_root_dir) / f"valid_{(index+1):03d}.png"

        image = Image.open(str(image_path)).convert("RGB")
        # image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)

        if self.transform:
            image = self.transform(image)

        # Text
        target = self.target_labels[index]

        target_length = [len(target)]
        target = torch.LongTensor(target)

        target_length = torch.LongTensor(target_length)

        return image, target, target_length

    def _generate_labels(self, text_raw_data):
        targets = []
        for line in text_raw_data:
            target = []
            for reading in line:
                if reading == " ":
                    target.append(self.space_index)
                elif reading in self.reading_to_signs:
                    target.extend(self.reading_to_signs[reading])
                else:
                    target.append(self.unk_index)
            targets.append(target)

        return targets
