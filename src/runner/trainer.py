import os
from pathlib import Path

import mlflow
import torch
import torch.optim as optim
from loguru import logger
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
        train_dataset: Dataset,
        valid_dataset: Dataset,
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

        # ignore 0 (Go Token)
        self._criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        self._optimizer = optim.Adam(
            self._model.parameters(), lr=0.003, weight_decay=0.000001
        )
        self._scaler = GradScaler()

        # setup dataset and dataloader
        self._train_loader = DataLoader(
            self._train_dataset, batch_size=self._cfg.batch_size, shuffle=True
        )
        self._valid_loader = DataLoader(
            self._valid_dataset, batch_size=self._cfg.batch_size, shuffle=True
        )
        self._best_loss = 999999999999
        self._train_iter = 1

        os.makedirs(self._cfg.dump.weight, exist_ok=True)
        self._last_weight = Path(self._cfg.dump.weight) / "last.pth"
        self._best_weight = Path(self._cfg.dump.weight) / "best.pth"

        mlflow.set_tracking_uri(self._cfg.dump.mlflow)

    def _iter(self, is_train: bool, images: torch.Tensor, targets: torch.Tensor):
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

        def _calc_loss(output, targets):
            output = output.view(-1, output.shape[-1])
            targets = targets[:, 1:].reshape(-1)  # without GO Token
            return self._criterion(output, targets)

        images = images.to(self._cfg.device)
        targets = targets.to(self._cfg.device)

        if is_train:  # ~~~ Training mode ~~~
            self._model.train()
            self._optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = self._model(images, targets[:, :-1])

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

        return float(loss.detach().cpu()), output, targets

    def _eval(self, iter_: int):
        epoch_valid_loss = 0
        for images, targets in tqdm(
            self._valid_loader, desc="Validation loop", dynamic_ncols=True, leave=False
        ):
            loss, batch_output, batch_targets = self._iter(
                is_train=False, images=images, targets=targets
            )
            epoch_valid_loss += loss

        mlflow.log_metric(
            "valid loss", epoch_valid_loss / len(self._valid_dataset), iter_
        )
        logger.info(
            "Valid loss : {}".format(epoch_valid_loss / len(self._valid_dataset))
        )
        batch_predict = batch_output.argmax(dim=2)
        for predict, target in zip(batch_predict, batch_targets):
            pred_decoded = self._train_dataset.decode(predict.tolist())
            # without GO Token
            target_decoded = self._train_dataset.decode(target[1:].tolist())
            logger.info("Predict[Valid]\t: {}".format(pred_decoded))
            logger.info("Target[Valid]\t: {}".format(target_decoded))

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

        train_loss = 0.0

        for images, targets in tqdm(
            self._train_loader,
            desc="EPOCH {} : train loop".format(curr_epoch),
            leave=False,
            dynamic_ncols=True,
        ):
            loss, batch_output, batch_targets = self._iter(
                is_train=True, images=images, targets=targets
            )
            train_loss += loss

            if self._train_iter % self._cfg.log_interval == 0:
                mlflow.log_metric(
                    "train loss",
                    train_loss / (self._cfg.log_interval * self._cfg.batch_size),
                    self._train_iter,
                )
                logger.info(
                    "Train loss : {}".format(
                        train_loss / (self._cfg.log_interval * self._cfg.batch_size)
                    )
                )
                batch_predict = batch_output.argmax(dim=2)
                for predict, target in zip(batch_predict, batch_targets):
                    pred_decoded = self._train_dataset.decode(predict.tolist())
                    # without GO TOKEN
                    target_decoded = self._train_dataset.decode(target[1:].tolist())
                    logger.info("Predict[Train]\t: {}".format(pred_decoded))
                    logger.info("Target[Train]\t: {}".format(target_decoded))

                self._eval(self._train_iter)
                train_loss = 0

            self._train_iter += 1

    def run(self):
        for i in range(self._cfg.epoch):
            self._epoch(i + 1)
