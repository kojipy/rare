import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import GradScaler
from omegaconf import DictConfig
import torch.nn.functional as F
import mlflow
from tqdm import tqdm

from ..dataset.dataset import SyntheticCuneiformLineImage


class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
        train_dataset: SyntheticCuneiformLineImage,
        valid_dataset: SyntheticCuneiformLineImage,
        model: torch.nn,
    ) -> None:
        self._cfg = cfg
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._model = model
        self._criterion = torch.nn.CrossEntropyLoss()
        self._optimizer = optim.Adam(
            self._model.parameters(), lr=0.0015, weight_decay=0.000001
        )
        self._scaler = GradScaler()

        # setup dataset and dataloader
        self._train_loader = DataLoader(
            self._train_dataset, batch_size=self._cfg.batch_size
        )
        self._valid_loader = DataLoader(
            self._valid_dataset, batch_size=self._cfg.batch_size
        )
        self._best_loss = 999999999999

        os.makedirs(self._cfg.dump.weight, exist_ok=True)
        self._last_weight = Path(self._cfg.dump.weight) / "last.pth"
        self._best_weight = Path(self._cfg.dump.weight) / "best.pth"

    def _epoch(self, curr_epoch: int):
        # training
        self._model.train()
        epoch_train_loss = 0

        for images, targets in tqdm(
            self._train_loader,
            desc="EPOCH {} : train loop".format(curr_epoch),
            leave=False,
            dynamic_ncols=True,
        ):

            images = images.to(self._cfg.device)
            targets = targets.to(self._cfg.device)

            self._optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                output = self._model(images, targets)
                targets = F.one_hot(targets, 190)

            loss = self._criterion(output, targets.type(torch.float))
            epoch_train_loss += loss

            loss.backward()
            self._optimizer.step()

        mlflow.log_metric("train loss", epoch_train_loss, curr_epoch)

        # evaluation
        self._model.eval()
        epoch_valid_loss = 0
        for images, targets in tqdm(
            self._valid_loader,
            desc="EPOCH {} : valid loop".format(curr_epoch),
            dynamic_ncols=True,
            leave=False,
        ):
            with torch.no_grad():
                images = images.to(self._cfg.device)
                targets = targets.to(self._cfg.device)

                with torch.cuda.amp.autocast():
                    output = self._model(images, targets)
                    targets = F.one_hot(targets, 190)
                    loss = self._criterion(output, targets.type(torch.float))

                epoch_valid_loss += loss

        mlflow.log_metric("valid loss", epoch_valid_loss, curr_epoch)

        torch.save(self._model.state_dict(), str(self._last_weight))
        if epoch_valid_loss < self._best_loss:
            torch.save(self._model.state_dict(), str(self._best_weight))
            self._best_loss = epoch_valid_loss

    def run(self):
        for i in range(self._cfg.epoch):
            self._epoch(i + 1)
