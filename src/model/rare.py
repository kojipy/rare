import torch
import torch.nn as nn
from omegaconf import DictConfig

from scipy.special import logsumexp

from .components.transformation import TPS_SpatialTransformerNetwork
from .components.feat_extraction import ResNet_FeatureExtractor
from .components.sequence_modeling import BidirectionalLSTM
from .components.prediction import Attention
from .lang_model import LitLstmLM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NINF = -1 * float("inf")


class Rare(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(Rare, self).__init__()
        self._cfg = cfg

        self._spn = TPS_SpatialTransformerNetwork(
            F=self._cfg.components.stn.num_fiducial,
            I_size=(self._cfg.components.stn.img_h, self._cfg.components.stn.img_w),
            I_r_size=(self._cfg.components.stn.img_h, self._cfg.components.stn.img_w),
            I_channel_num=self._cfg.rare.input_channel,
        )

        self._feat_ext = ResNet_FeatureExtractor(
            self._cfg.rare.input_channel, self._cfg.rare.output_channel
        )

        # Transform final (img_h/16-1) -> 1
        self._pool = nn.AdaptiveAvgPool2d((None, 1))

        self._srn = nn.Sequential(
            BidirectionalLSTM(
                self._cfg.rare.output_channel,
                self._cfg.components.bilstm.hidden_size,
                self._cfg.components.bilstm.hidden_size,
            ),
            BidirectionalLSTM(
                self._cfg.components.bilstm.hidden_size,
                self._cfg.components.bilstm.hidden_size,
                self._cfg.components.bilstm.hidden_size,
            ),
        )

        self._prediction = Attention(
            self._cfg.components.bilstm.hidden_size,
            self._cfg.components.bilstm.hidden_size,
            self._cfg.rare.num_class,
        )

    def forward(self, img, text, is_train=True):

        rectified = self._spn(img)

        visual_feature = self._feat_ext(rectified)
        visual_feature = self._pool(
            visual_feature.permute(0, 3, 1, 2)
        )  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        contextual_feature = self._srn(visual_feature)

        prediction = self._prediction(
            contextual_feature.contiguous(),
            text,
            is_train,
            batch_max_length=self._cfg.rare.label_max_length,
        )

        return prediction

    def predict_with_lm(self, img, text, lm: LitLstmLM):
        pred = self.forward(img, text, is_train=False)
        pred = pred.squeeze()

        beams = [([], 0)]  # (prefix, accumulated_log_prob)
        for t in range(self._cfg.rare.label_max_length):
            new_beams = []
            for prefix, accumulated_log_prob in beams:
                for c in range(self._cfg.rare.num_class):
                    log_prob = pred[t, c]
                    if log_prob < self._cfg.components.beamsearch.emission_thresh:
                        continue
                    new_prefix = prefix + [c]
                    # log(p1 * p2) = log_p1 + log_p2
                    new_accu_log_prob = accumulated_log_prob + log_prob
                    new_beams.append((new_prefix, new_accu_log_prob))

            # sorted by accumulated_log_prob
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[: self._cfg.components.beamsearch.beam_size]

        # sum up beams to produce labels
        total_accu_log_prob = {}
        for labels, accu_log_prob in beams:
            labels = tuple(labels)
            # log(p1 + p2) = logsumexp([log_p1, log_p2])
            total_accu_log_prob[labels] = logsumexp(
                [
                    accu_log_prob.detach().cpu().numpy(),
                    total_accu_log_prob.get(labels, NINF),
                ]
            )

        labels_beams = []
        for labels, accu_log_prob in total_accu_log_prob.items():
            language_model_x = torch.Tensor([labels[:-1]]).to(torch.int).to(device)
            language_model_y = torch.Tensor([labels[1:]]).to(torch.int).to(device)
            language_model_output = lm(language_model_x)
            vocab_size = language_model_output.size()[-1]
            lanuage_model_prob = torch.nn.functional.nll_loss(
                torch.nn.functional.log_softmax(language_model_output, dim=-1)
                .contiguous()
                .view(-1, vocab_size),
                language_model_y.contiguous().view(-1).to(torch.long),
                reduction="sum",
                ignore_index=lm.pad,
            )
            print(labels, accu_log_prob, lanuage_model_prob)
            accu_log_prob += (
                self._cfg.components.lmfusion.weight * lanuage_model_prob.item()
            )
            labels_beams.append((list(labels), accu_log_prob))
        labels_beams.sort(key=lambda x: x[1], reverse=True)
        labels = labels_beams[0][0]

        return labels