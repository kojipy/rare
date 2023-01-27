import numbers

import torchvision.transforms as T
from omegaconf import OmegaConf

from src.dataset.dataset import SyntheticCuneiformLineImage
from src.dataset.valid_dataset import SyntheticCuneiformValidationLineImage
from src.model.rare import Rare
from src.runner.trainer import Trainer

transform = T.Compose(
    [
        # T.ToTensor(),
        T.Pad(padding=(0, 16)),
        T.RandomApply(
            transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.3
        ),
        T.RandomRotation(degrees=(0, 3)),
        T.RandomApply(transforms=[T.AugMix()], p=0.3),
        T.Resize(96),
        # T.Grayscale(),
        T.ToTensor(),
    ]
)


def get_padding(image):
    max_w = 1344
    max_h = 64

    imsize = image.size
    h_padding = max_w - imsize[0]
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5

    padding = (0, 0, h_padding, 0)

    return padding


class NewPad:
    def __init__(self, fill=0, padding_mode="constant"):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ["constant", "edge", "reflect", "symmetric"]

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return T.functional.pad(img, get_padding(img), self.fill, self.padding_mode)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(padding={0}, fill={1}, padding_mode={2})".format(
                self.fill, self.padding_mode
            )
        )


validation_transform = T.Compose(
    [
        T.Resize(96),
        NewPad(),
        #    T.CenterCrop((64, 1536)),
        T.ToTensor(),
        # T.Grayscale(),
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

    # valid_dataset = SyntheticCuneiformLineImage(
    #     target_signs_file_path=cfg.dataset.signs,
    #     images_root_dir=cfg.dataset.valid.image,
    #     texts_root_dir=cfg.dataset.valid.label,
    #     label_max_length=cfg.rare.label_max_length,
    #     first_idx=cfg.dataset.valid.first_index,
    #     last_idx=cfg.dataset.valid.last_index,
    #     transform=transform,
    # )

    valid_dataset = SyntheticCuneiformValidationLineImage(
        target_signs_file_path=cfg.dataset.signs,
        images_root_dir=cfg.dataset.valid.image,
        label_max_length=cfg.rare.label_max_length,
        transform=validation_transform,
    )

    trainer = Trainer(cfg, train_dataset, valid_dataset, rare)
    trainer.run()
