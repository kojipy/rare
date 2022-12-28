import torchvision.transforms as T

from dataset.dataset import SyntheticCuneiformLineImage
from model.rare import Rare


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

    dataset = SyntheticCuneiformLineImage(
        target_signs_file_path="dataset/target_hittite_cuneiform_signs.json",
        images_root_dir="dataset/images",
        texts_root_dir="dataset/annotations",
        first_idx=0,
        last_idx=96 - 1,
        transform=transform,
    )
