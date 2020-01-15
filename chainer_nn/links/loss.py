import chainer
import chainer.functions as F
from chainer.links.rnn.n_step_rnn import argsort_list_descent
from chainer.links.rnn.n_step_rnn import permutate_list


class CRF1d(chainer.link.Link):

    def __init__(self, n_label, initial_cost=None):
        super().__init__()
        if initial_cost is None:
            initial_cost = chainer.initializers.constant.Zero()

        with self.init_scope():
            self.cost = chainer.Parameter(
                initializer=initial_cost, shape=(n_label, n_label))

    def forward(self, xs, ys, reduce='mean'):
        indices = argsort_list_descent(xs)
        xs = permutate_list(xs, indices, inv=False)
        xs = F.transpose_sequence(xs)
        ys = permutate_list(ys, indices, inv=False)
        ys = F.transpose_sequence(ys)
        return F.crf1d(self.cost, xs, ys, reduce)

    def argmax(self, xs):
        return argmax_crf1d(self.cost, xs)


def argmax_crf1d(cost, xs):
    indices = argsort_list_descent(xs)
    xs = permutate_list(xs, indices, inv=False)
    xs = F.transpose_sequence(xs)
    score, path = F.argmax_crf1d(cost, xs)
    path = F.transpose_sequence(path)
    path = permutate_list(path, indices, inv=True)
    score = F.permutate(score, indices, inv=True)
    return score, path
