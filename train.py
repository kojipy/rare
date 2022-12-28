from omegaconf import OmegaConf
import torchvision.transforms as T

from src.model.rare import Rare
from src.dataset.dataset import SyntheticCuneiformLineImage
from src.runner.trainer import Trainer


transform = T.Compose(
    [
        #           T.ToTensor(),
        #           T.Pad(padding=(0, 16)),
        T.RandomApply(
            transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.3
        ),
        T.RandomRotation(degrees=(0, 3)),
        #            T.RandomApply(
        #                transforms=[T.AugMix()],
        #                p=0.3
        #            ),
        T.Resize(96),
        #            T.Grayscale(),
        T.ToTensor(),
    ]
)

if __name__ == "__main__":
    model_cfg = OmegaConf.load("config/rare.yaml")
    train_cfg = OmegaConf.load("config/train.yaml")
    cfg = OmegaConf.merge(model_cfg, train_cfg)

    rare = Rare(cfg).to(cfg.device)

    train_dataset = SyntheticCuneiformLineImage(
        target_signs_file_path=cfg.dataset.signs,
        images_root_dir=cfg.dataset.train.image,
        texts_root_dir=cfg.dataset.train.label,
        label_max_length=cfg.rare.label_max_length,
        first_idx=cfg.dataset.train.first_index,
        last_idx=cfg.dataset.train.last_index,
        transform=transform,
    )

    valid_dataset = SyntheticCuneiformLineImage(
        target_signs_file_path=cfg.dataset.signs,
        images_root_dir=cfg.dataset.valid.image,
        texts_root_dir=cfg.dataset.valid.label,
        label_max_length=cfg.rare.label_max_length,
        first_idx=cfg.dataset.valid.first_index,
        last_idx=cfg.dataset.valid.last_index,
        transform=transform,
    )

    trainer = Trainer(cfg, train_dataset, valid_dataset, rare)
    trainer.run()
