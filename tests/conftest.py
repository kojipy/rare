import pytest
from omegaconf import OmegaConf


@pytest.fixture
def config():
    model_cfg = OmegaConf.load("tests/assets/config/rare.yaml")
    train_cfg = OmegaConf.load("tests/assets/config/train.yaml")
    cfg = OmegaConf.merge(model_cfg, train_cfg)
    return cfg
