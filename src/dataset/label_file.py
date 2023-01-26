import json
from typing import List


class LabelFile:
    def __init__(self, path: str) -> None:
        self._path = path
        self._text = self._load()

    @property
    def text(self) -> List[str]:
        return self._text

    def _load(self) -> List[str]:
        with open(self._path) as f:
            data = json.load(f)

        target = []
        for line in data["line"]:
            for words in line["signs"]:
                for reading_dict in words:
                    reading = reading_dict["reading"]
                    target.append(reading)
                target.append(" ")

        target = target[:-1]  # remove last space

        return target
