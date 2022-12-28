from src.dataset.dataset import SyntheticCuneiformLineImage


def test_load(config):
    dataset = SyntheticCuneiformLineImage(
        cfg=config,
        target_signs_file_path="tests/assets/target_hittite_cuneiform_signs.json",
        images_root_dir="tests/assets/images",
        texts_root_dir="tests/assets/annotations",
        first_idx=0,
        last_idx=2,
    )

    dataset.__getitem__(0)
