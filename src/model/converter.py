from typing import List

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LabelConverter:
    def __init__(self, label_max_length: int):
        self._label_max_len = label_max_length

    def encode(self, text: List[int]):
        text = torch.tensor(text)
        text_with_pad = torch.zeros(self._label_max_len + 1, dtype=torch.long)
        text_with_pad[: len(text)] = text
        return text_with_pad
