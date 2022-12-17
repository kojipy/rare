import pytorch_lightning as pl
import torch


class LitLstmLM(pl.LightningModule):
    def __init__(self, vocab: dict, dim_emb=128, dim_hid=256):
        super().__init__()
        self.vocab = vocab
        self.pad = vocab["<pad>"]
        self.embed = torch.nn.Embedding(len(vocab), dim_emb)
        self.rnn = torch.nn.LSTM(dim_emb, dim_hid, batch_first=True)
        self.out = torch.nn.Linear(dim_hid, len(vocab))

    def forward(self, x):
        x = self.embed(x)
        x, (h, c) = self.rnn(x)
        x = self.out(x)
        return x
