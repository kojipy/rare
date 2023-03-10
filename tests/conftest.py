import pytest
from omegaconf import OmegaConf

from src.model.lang_model import LitLstmLM4Debug
from src.model.rare import Rare
from src.model.converter import LabelConverter


@pytest.fixture
def config():
    model_cfg = OmegaConf.load("tests/assets/config/rare.yaml")
    train_cfg = OmegaConf.load("tests/assets/config/train.yaml")
    cfg = OmegaConf.merge(model_cfg, train_cfg)
    return cfg


@pytest.fixture
def rare(config):
    return Rare(config)


@pytest.fixture
def lang_model():
    return LitLstmLM4Debug()


@pytest.fixture
def converter():
    signs_json = "tests/assets/target_hittite_cuneiform_signs.json"
    return LabelConverter(30, signs_json)
