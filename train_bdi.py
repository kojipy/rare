"""
Training RARE with Born-Digital-Images dataset.
dataset URL : https://rrc.cvc.uab.es/?ch=1 
"""
import torchvision.transforms as T
from omegaconf import OmegaConf

from src.dataset.bdi_dataset import BdiDataset
from src.model.rare import Rare
from src.runner.trainer import Trainer

transform = T.Compose(
    [
        #           T.ToTensor(),
        #           T.Pad(padding=(0, 16)),
        # T.RandomApply(
        #     transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.3
        # ),
        # T.RandomRotation(degrees=(0, 3)),
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
    train_dataset = BdiDataset(
        root_dir="dataset/born-digital-images",
        first_index=1,
        last_index=3000,
        label_max_length=30,
        img_height=96,
        img_width=1536,
        transform=transform,
    )

    valid_dataset = BdiDataset(
        root_dir="dataset/born-digital-images",
        first_index=3001,
        last_index=3500,
        label_max_length=30,
        img_height=96,
        img_width=1536,
        transform=transform,
    )

    model_cfg = OmegaConf.load("config/rare.yaml")
    train_cfg = OmegaConf.load("config/train.yaml")
    cfg = OmegaConf.merge(model_cfg, train_cfg)
    cfg.rare.num_classes = train_dataset.num_classes

    rare = Rare(cfg).to(cfg.device)

    trainer = Trainer(cfg, train_dataset, valid_dataset, rare)
    trainer.run()
