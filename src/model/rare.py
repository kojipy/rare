"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn
from omegaconf import DictConfig

from .components.transformation import TPS_SpatialTransformerNetwork
from .components.feat_extraction import ResNet_FeatureExtractor
from .components.sequence_modeling import BidirectionalLSTM
from .components.prediction import Attention
from .lang_model import LitLstmLM


class Rare(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(Rare, self).__init__()
        self._cfg = cfg
        self._stn_cfg = self._cfg.components.stn
        self._bilstm_cfg = self._cfg.components.bilstm
        self._rare_cfg = self._cfg.rare

        self._spn = TPS_SpatialTransformerNetwork(
            F=self._stn_cfg.num_fiducial,
            I_size=(self._stn_cfg.img_h, self._stn_cfg.img_w),
            I_r_size=(self._stn_cfg.img_h, self._stn_cfg.img_w),
            I_channel_num=self._rare_cfg.input_channel,
        )

        self._feat_ext = ResNet_FeatureExtractor(
            self._rare_cfg.input_channel, self._rare_cfg.output_channel
        )

        # Transform final (img_h/16-1) -> 1
        self._pool = nn.AdaptiveAvgPool2d((None, 1))

        self._srn = nn.Sequential(
            BidirectionalLSTM(
                self._rare_cfg.output_channel,
                self._bilstm_cfg.hidden_size,
                self._bilstm_cfg.hidden_size,
            ),
            BidirectionalLSTM(
                self._bilstm_cfg.hidden_size,
                self._bilstm_cfg.hidden_size,
                self._bilstm_cfg.hidden_size,
            ),
        )

        self._prediction = Attention(
            self._bilstm_cfg.hidden_size,
            self._bilstm_cfg.hidden_size,
            self._rare_cfg.num_class,
        )

    def forward(self, input, text, is_train=True):

        input = self._spn(input)

        visual_feature = self._feat_ext(input)
        visual_feature = self._pool(
            visual_feature.permute(0, 3, 1, 2)
        )  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        contextual_feature = self._srn(visual_feature)

        prediction = self._prediction(
            contextual_feature.contiguous(),
            text,
            is_train,
            batch_max_length=self._rare_cfg.label_max_length,
        )

        return prediction

    def predict_with_lm(self, lm: LitLstmLM):
        pass
