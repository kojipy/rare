import torch
from omegaconf import DictConfig

from torch.utils.data import DataLoader

from src.dataset.dataset import SyntheticCuneiformLineImage
from src.model.converter import LabelConverter


class Evaluator:
    """
    Evaluate accuracy of the model using with test dataset.
    """

    def __init__(
        self,
        cfg: DictConfig,
        test_dataset: SyntheticCuneiformLineImage,
        model: torch.nn,
    ) -> None:
        self._cfg = cfg
        self._test_dataset = test_dataset
        self._loader = DataLoader(self._test_dataset, batch_size=1)
        self._converter = LabelConverter(cfg.rare.label_max_length, cfg.dataset.signs)

        model.load_state_dict(torch.load(cfg.weight, map_location=cfg.device))
        model.to(cfg.device)
        self._model = model

    def run(self):
        """
        Main Entory point of this class. main process of this class will start.
        """
        self._model.eval()
        for images, targets in self._loader:
            with torch.no_grad():
                images = images.to(self._cfg.device)
                targets = targets.to(self._cfg.device)

                with torch.cuda.amp.autocast():
                    output = self._model.predict(images)

                indecies = output.squeeze().cpu().numpy()
                chars = self._converter.decode(indecies)
                print(chars)

    def dump(self):
        pass
