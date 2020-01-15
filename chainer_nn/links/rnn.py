import chainer

from chainer_nn.functions import rnn


class NStepRNNBase(chainer.links.rnn.n_step_rnn.NStepRNNBase):

    def __init__(self, n_layers, in_size, out_size, dropout,
                 recurrent_dropout=0.0, use_variational_dropout=False,
                 **kwargs):
        self.recurrent_dropout = recurrent_dropout
        self.use_variational_dropout = use_variational_dropout
        super().__init__(n_layers, in_size, out_size, dropout, **kwargs)


class NStepLSTMBase(NStepRNNBase):
    n_weights = 8

    def forward(self, hx, cx, xs, **kwargs):
        (hy, cy), ys = self._call([hx, cx], xs, **kwargs)
        return hy, cy, ys


class NStepLSTM(NStepLSTMBase):
    use_bi_direction = False

    def rnn(self, *args):
        assert len(args) == 7
        return rnn.n_step_lstm(
            *args, self.recurrent_dropout, self.use_variational_dropout)

    @property
    def n_cells(self):
        return 2


class NStepBiLSTM(NStepLSTMBase):
    use_bi_direction = True

    def rnn(self, *args):
        assert len(args) == 7
        return rnn.n_step_bilstm(
            *args, self.recurrent_dropout, self.use_variational_dropout)

    @property
    def n_cells(self):
        return 2
