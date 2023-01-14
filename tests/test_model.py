import torch
from omegaconf import OmegaConf

from src.model.rare import Rare


def test_inference():
    cfg = OmegaConf.load("config/rare.yaml")
    image = torch.zeros((1, 3, 96, 2304))
    rare = Rare(cfg)
    rare.eval()
    rare(image)


def test_inference_lm(rare, lang_model):
    rare = rare
    lang_model = lang_model

    rare.eval()
    lang_model.eval()

    img = torch.zeros(1, 3, 96, 2304)
    _ = rare.predict_with_lm(img, lang_model)
