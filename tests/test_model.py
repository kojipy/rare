import torch
from omegaconf import OmegaConf

from src.model.rare import Rare


def test_inference():
    cfg = OmegaConf.load("config/rare.yaml")
    image = torch.zeros((1, 3, 96, 2304)).to("cuda")
    rare = Rare(cfg).to("cuda")
    rare.eval()
    rare(image)
