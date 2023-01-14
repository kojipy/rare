import os
from pathlib import Path

import mlflow
import torch
import torch.optim as optim
from loguru import logger
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..dataset.dataset import SyntheticCuneiformLineImage
from .loss import CustomCrossEntropy


class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
        train_dataset: SyntheticCuneiformLineImage,
        valid_dataset: SyntheticCuneiformLineImage,
        model: torch.nn,
    ) -> None:
        logger.info(cfg)

        self._cfg = cfg
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._model = model
        if cfg.weight is not None:
            self._model.load_state_dict(torch.load(cfg.weight, map_location=cfg.device))
            logger.info("Model file : {} : is loaded".format(cfg.weight))

        self._criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        # self._criterion = CustomCrossEntropy()

        self._optimizer = optim.Adam(
            self._model.parameters(), lr=0.0015, weight_decay=0.000001
        )
        self._scaler = GradScaler()

        # setup dataset and dataloader
        self._train_loader = DataLoader(
            self._train_dataset, batch_size=self._cfg.batch_size, shuffle=True
        )
        self._valid_loader = DataLoader(
            self._valid_dataset, batch_size=self._cfg.batch_size
        )
        self._best_loss = 999999999999
        self._train_iter = 1

        os.makedirs(self._cfg.dump.weight, exist_ok=True)
        self._last_weight = Path(self._cfg.dump.weight) / "last.pth"
        self._best_weight = Path(self._cfg.dump.weight) / "best.pth"

        mlflow.set_tracking_uri(self._cfg.dump.mlflow)

    def _iter(
        self, is_train: bool, images: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """
        This method recievs input X, and ground truth Y. after that run model inference.
        If this function is called as train mode, call backward.

        Args:
            is_train (bool): True if train mode, else False.
            images (torch.Tensor): model input X.
            targets (torch.Tensor): ground truth Y.

        Returns:
            loss (float): loss between model output and targets.
        """

        def _calc_loss(output, targets) -> float:
            output = output.view(-1, output.shape[-1])
            targets = targets.view(-1)
            return self._criterion(output, targets)

        images = images.to(self._cfg.device)
        targets = targets.to(self._cfg.device)

        if is_train:  # ~~~ Training mode ~~~
            self._model.train()
            self._optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = self._model(images, targets)

            loss = _calc_loss(output, targets)

            self._scaler.scale(loss).backward()
            self._scaler.step(self._optimizer)
            self._scaler.update()

        else:  # ~~~ Validation mode ~~~
            self._model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output = self._model(images)

                loss = _calc_loss(output, targets)

        return float(loss.detach().cpu())

    def _eval(self, iter_: int):
        epoch_valid_loss = 0
        for images, targets in tqdm(
            self._valid_loader, desc="Validation loop", dynamic_ncols=True, leave=False
        ):
            loss = self._iter(is_train=False, images=images, targets=targets)
            epoch_valid_loss += loss

        mlflow.log_metric(
            "valid loss", epoch_valid_loss / len(self._valid_dataset), iter_
        )

        self._save(epoch_valid_loss)

    def _save(self, eval_loss: float):
        """
        Save model weight file. `last.pth` file is alawys saved. `best.pth` file is
        saved only if model is best accuracy.
        """
        torch.save(self._model.state_dict(), str(self._last_weight))
        if eval_loss < self._best_loss:
            torch.save(self._model.state_dict(), str(self._best_weight))
            self._best_loss = eval_loss

    def _epoch(self, curr_epoch: int):
        """
        This function runs one epoch.
        Args:
            curr_epoch (int): current epoch (starts from 1, not 0)
        """

        train_loss = 0

        for images, targets in tqdm(
            self._train_loader,
            desc="EPOCH {} : train loop".format(curr_epoch),
            leave=False,
            dynamic_ncols=True,
        ):
            loss = self._iter(is_train=True, images=images, targets=targets)
            train_loss += loss

            if self._train_iter % self._cfg.log_interval == 0:
                mlflow.log_metric(
                    "train loss",
                    train_loss / (self._cfg.log_interval * self._cfg.batch_size),
                    self._train_iter,
                )
                self._eval(self._train_iter)
                train_loss = 0

            self._train_iter += 1

    def run(self):
        for i in range(self._cfg.epoch):
            self._epoch(i + 1)
