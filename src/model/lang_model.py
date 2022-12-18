import torch.nn as nn


class LitLstmLM(nn.Module):
    def __init__(self, vocab: dict, dim_emb=128, dim_hid=256):
        super().__init__()
        self.vocab = vocab
        self.pad = vocab["<pad>"]
        self.embed = nn.Embedding(len(vocab), dim_emb)
        self.rnn = nn.LSTM(dim_emb, dim_hid, batch_first=True)
        self.out = nn.Linear(dim_hid, len(vocab))

    def forward(self, x):
        x = self.embed(x)
        x, (h, c) = self.rnn(x)
        x = self.out(x)
        return x


class LitLstmLM4Debug(nn.Module):
    def __init__(self, vocab: int = 190, dim_emb=128, dim_hid=256):
        super().__init__()
        self.vocab = vocab
        self.pad = 0
        self.embed = nn.Embedding(vocab, dim_emb)
        self.rnn = nn.LSTM(dim_emb, dim_hid, batch_first=True)
        self.out = nn.Linear(dim_hid, vocab)

    def forward(self, x):
        x = self.embed(x)
        x, (h, c) = self.rnn(x)
        x = self.out(x)
        return x
