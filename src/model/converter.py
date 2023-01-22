import json
from typing import Dict, List

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LabelConverter:
    def __init__(self, label_max_length: int, target_signs_json: str):
        self._label_max_len = label_max_length
        self._reading_to_signs = {}
        self._PAD_TOKEN = "[PAD]"
        self._GO_TOKEN = "[GO]"
        self._STOP_TOKEN = "[STOP]"
        self._sign_to_index = {
            self._PAD_TOKEN: 0,
            self._GO_TOKEN: 1,
            self._STOP_TOKEN: 2,
        }
        self._index_to_sign = {
            0: self._PAD_TOKEN,
            1: self._GO_TOKEN,
            2: self._STOP_TOKEN,
        }
        # self._space_index = None
        self._unk_index = None

        self._load_target_signs(target_signs_json)

    @property
    def reading2signs(self):
        return self._reading_to_signs

    @property
    def sign2index(self):
        return self._sign_to_index

    def decode(self, text: List[int]) -> List[str]:
        """
        Decode list of indicies to text.

        Args:
            text (str): list of indiceis to be decoded

        Returns:
            List[str]: decoded string
        """
        decoded_chars = []
        for label in text:
            if self._sign_to_index[self._STOP_TOKEN] == label:
                break
            if self._unk_index == label:
                decoded_chars.append("UNK")
                continue
            # if self._space_index == label:
            # decoded_chars.append(" ")
            # continue
            if self._PAD_TOKEN == label:
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
        target = [self._sign_to_index[self._GO_TOKEN]]  # 1st index is for GO TOKEN
        for line in text["line"]:
            for words in line["signs"]:
                for reading_dict in words:
                    reading = reading_dict["reading"]
                    if reading in self._reading_to_signs:
                        target.extend(self._reading_to_signs[reading])
                    else:
                        target.append(self._unk_index)
                # target.append(self._space_index)

        # target = target[:-1]  # remove last space
        target.append(self._sign_to_index[self._STOP_TOKEN])  # end with STOP TOKEN
        text = torch.tensor(target)
        text_with_pad = torch.zeros(self._label_max_len + 1, dtype=torch.long)
        text_with_pad[: len(text)] = text
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
                    # 0 is for PAD TOKEN 1 is for GO TOKEN, 2 is for STOP TOKEN
                    idx: int = len(self._sign_to_index)
                    self._sign_to_index[sign] = idx
                    self._index_to_sign[idx] = sign

                sign_indices.append(self._sign_to_index[sign])

            for reading in loaded[signs]["readings"]:
                self._reading_to_signs[reading["reading"]] = sign_indices

        # self._space_index = len(self._sign_to_index)
        self._unk_index = len(self._sign_to_index)

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
                # target.append(self._space_index)

        return target
