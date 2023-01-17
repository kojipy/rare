import json
from typing import Dict, List

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LabelConverter:
    def __init__(self, label_max_length: int, target_signs_json: str):
        self._label_max_len = label_max_length
        self._reading_to_signs = {}
        self._sign_to_index = {}
        self._index_to_sign = {}
        self._space_index = None
        self._unk_index = None
        self._pad_index = None
        self._load_target_signs(target_signs_json)

    @property
    def reading2signs(self):
        return self._reading_to_signs

    @property
    def sign2index(self):
        return self._sign_to_index

    def decode(self, text: List[int]) -> str:
        """
        Decode list of indicies to text.

        Args:
            text (str): list of indiceis to be decoded

        Returns:
            str: decoded string
        """
        decoded_chars = []
        for label in text:
            if self._unk_index == label:
                decoded_chars.append("UNK")
                continue
            if self._space_index == label:
                decoded_chars.append(" ")
                continue
            if 0 == label:  # 0 is index for space
                continue
            decoded_chars.append(self._index_to_sign[label])

        return decoded_chars

    def encode(self, text: Dict) -> torch.Tensor:
        """
        Encode string to list of index of character

        Args:
            text (Dict): loaded data from annotation json file.

        Returns:
            List[int]: list of indecies for each character.
        """
        target = []
        for line in text["line"]:
            for words in line["signs"]:
                for reading_dict in words:
                    reading = reading_dict["reading"]
                    if reading in self._reading_to_signs:
                        target.extend(self._reading_to_signs[reading])
                    else:
                        target.append(self._unk_index)
                target.append(self._space_index)

        text = target[:-1]  # remove last space
        text = torch.tensor(text)
        text_with_pad = torch.ones(self._pad_index, dtype=torch.long)
        text_with_pad[1 : len(text) + 1] = text  # first index is for GO TOKEN
        return text_with_pad

    def _load_target_signs(self, target_signs_file_path: str):
        """
        Load json file of signs.

        Args:
            target_signs_file_path (str): path of signs json file.

        Returns:
            None
        """

        with open(target_signs_file_path) as f:
            loaded = json.load(f)

        for signs in sorted(loaded):
            sign_indices = []  # list of int sign indices
            for sign in signs.split("."):
                if sign not in self._sign_to_index:
                    idx: int = len(self._sign_to_index) + 1  # 0 is for blank
                    self._sign_to_index[sign] = idx
                    self._index_to_sign[idx] = sign

                sign_indices.append(self._sign_to_index[sign])

            for reading in loaded[signs]["readings"]:
                self._reading_to_signs[reading["reading"]] = sign_indices

        self._space_index = len(self._sign_to_index)
        self._unk_index = len(self._sign_to_index) + 1
        self._pad_index = len(self._sign_to_index) + 2

    def _load_text(self, path) -> List[int]:
        """
        Read annotation json file and return list of indecies of character.

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
                    if reading in self._reading_to_signs:
                        target.extend(self._reading_to_signs[reading])
                    else:
                        target.append(self._unk_index)
                target.append(self._space_index)

        return target[:-1]  # remove last space
