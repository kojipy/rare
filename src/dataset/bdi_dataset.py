"""
This script is implementation for Born Digital Images dataset.
"""
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .dataset import my_resize


class BdiDataset(Dataset):
    _gt_path_pattern = "word_{}.png"  # image file naming rule of this dataset.

    def __init__(
        self,
        root_dir: str,
        first_index: int,
        last_index: int,
        label_max_length: int,
        img_height: int = 96,
        img_width: int = 64 * 24,
        transform=None,
    ) -> None:
        super().__init__()
        assert first_index != 0

        self._root_dir = Path(root_dir)
        self._gt_file = self._root_dir / "gt.txt"
        self._first_index = first_index  # 1 start index
        self._last_index = last_index  # 1 start index
        self._labels: Dict = {}
        self._label_max_length = label_max_length
        self.img_height = img_height
        self.img_width = img_width
        self._transform = transform

        self._char2idx: Dict[str, int] = {}
        self._idx2char: Dict[str, int] = {}
        self.__load_gt()

    @property
    def num_classes(self) -> int:
        return len(self._char2idx) + 1

    def __load_gt(self):
        """
        Load GroundTruth text file. While parsing text file, this method do two procces.
         - First, Generate image path and label pairs.s
         - Secound, collects unique characters appears in GroundTruth text file that
        will be `self._classes` property of this class.
        """
        with open(str(self._gt_file), encoding="utf-8-sig") as f:
            lines = f.read().splitlines()

        chars_uniq = set()
        for line in lines:
            imgname, label = line.split(".png, ")
            imgname += ".png"
            self._labels[str(self._root_dir / imgname)] = label[1:-1]  # remove `"`
            for char in label:
                chars_uniq.add(char)

        classes = list(chars_uniq)
        classes.sort()
        for char in classes:
            self._char2idx[char] = classes.index(char) + 1  # 0 index is for padding

        # create inverted dict of self._char2idx
        for char_key in self._char2idx:
            idx = self._char2idx[char_key]
            self._idx2char[idx] = char_key

    def __len__(self) -> int:
        return self._last_index - self._first_index + 1

    def __idx2path(self, index: int) -> str:
        """
        This method translate index to image path.
        """
        real_index = self._first_index + index
        imgname = self._gt_path_pattern.format(real_index)
        path = str(self._root_dir / imgname)
        return path

    def __getitem__(self, index: int):
        img_path = self.__idx2path(index)
        image = Image.open(img_path).convert("RGB")

        width = int(image.width * (self.img_height / image.height))
        width = my_resize(width, 128, 1536)

        image = image.resize((width, self.img_height), resample=Image.BILINEAR)
        image = ImageOps.pad(
            image, (self.img_width, self.img_height), color=(0, 0, 0), centering=(0, 0)
        )

        if self._transform:
            image = self._transform(image)

        label = self._labels[img_path]
        target = self.encode(label)
        target = torch.tensor(target, dtype=torch.long)

        return image, target

    def encode(self, label: str) -> List[int]:
        """
        Translate text label to list of class index.
        """
        encoded = [0] * (self._label_max_length + 1)
        for i, char in enumerate(label):
            cls_idx = self._char2idx[char]
            encoded[i] = cls_idx
        return encoded

    def decode(self, text: List[int]):
        decoded_chars = []
        for idx in text:
            if 0 == idx:  # 0 is index for space
                continue
            decoded_chars.append(self._idx2char[idx])
        return decoded_chars
