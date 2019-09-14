import chainer
import chainer.functions as F
import numpy as np

from chainer_nn.functions.noise import dropout as apply_dropout
from chainer_nn.links.rnn import NStepLSTM


class CharLSTM(chainer.link.Chain):

    def __init__(self, char_embeddings, out_size,
                 char_dropout, lstm_dropout, recurrent_dropout=0.0):
        super().__init__()
        vocab_size, embed_size = char_embeddings.shape
        with self.init_scope():
            self.embed = chainer.links.EmbedID(
                vocab_size, embed_size, char_embeddings)
            self.lstm = NStepLSTM(
                1, embed_size, out_size, lstm_dropout, recurrent_dropout)
        self.char_dropout = char_dropout

    def forward(self, xs):
        offsets = np.array([len(x) for x in xs[:-1]]).cumsum()
        xs = apply_dropout(self.embed(self.xp.hstack(xs)), self.char_dropout)
        xs = F.split_axis(xs, offsets, axis=0)
        ys = F.squeeze(self.lstm(hx=None, cx=None, xs=xs)[0], axis=0)
        return ys


class CharCNN(chainer.link.Chain):

    def __init__(self, char_embeddings, pad_id, out_size, window_size, dropout,
                 nobias=False, initialW=None, initial_bias=None):
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd value: '{}' is given"
                             .format(window_size))
        super().__init__()
        char_vocab_size, char_embed_size = char_embeddings.shape
        with self.init_scope():
            self.embed = chainer.links.EmbedID(
                in_size=char_vocab_size,
                out_size=char_embed_size,
                initialW=char_embeddings)
            self.conv = chainer.links.Convolution2D(
                in_channels=1,
                out_channels=out_size,
                ksize=(window_size, char_embed_size),
                stride=(1, char_embed_size),
                pad=(window_size // 2, 0),
                nobias=nobias,
                initialW=initialW,
                initial_bias=initial_bias)
        self.out_size = out_size
        self._pad_id = pad_id
        self._pad_width = window_size // 2
        self._padding = np.array([pad_id] * self._pad_width, dtype=np.int32)
        self._dropout = dropout

    def forward(self, chars):
        xp = self.xp
        if not isinstance(chars, (tuple, list)):
            chars = [chars]
        lengths = [len(w) for w in chars]
        n_words = len(lengths)
        pad_width = self._pad_width

        char_ids = np.full(
            (len(chars), max(lengths) + pad_width * 2), self._pad_id, np.int32)
        mask = np.full(char_ids.shape, np.inf, np.float32)
        for i, (w, length) in enumerate(zip(chars, lengths)):
            char_ids[i, pad_width:pad_width + length] = w
            mask[i, pad_width:pad_width + length] = 0.
        mask = xp.expand_dims(xp.array(mask), axis=2)

        xs = self.embed(xp.array(char_ids))
        xs = apply_dropout(xs, self._dropout)
        C = self.conv(F.expand_dims(xs, axis=1))
        C = F.transpose(F.squeeze(C, axis=3), (0, 2, 1))
        assert C.shape == (n_words,
                           pad_width + max(lengths) + pad_width,
                           self.out_size)
        ys = F.max(C - mask, axis=1)
        return ys
