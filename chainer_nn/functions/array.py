import chainer
import chainer.functions as F
from chainer.utils import type_check


def unpad_sequence(x, lengths):
    return UnpadSequence(lengths).apply((x,))


class UnpadSequence(chainer.function_node.FunctionNode):

    def __init__(self, lengths):
        self.lengths = lengths

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].ndim > 0)
        type_check.expect(in_types[0].shape[0] == len(self.lengths))
        max_length = type_check.make_variable(max(self.lengths), 'max_length')
        type_check.expect(in_types[0].shape[1] >= max_length)

    def forward(self, xs):
        if xs[0].size == 0:
            ys = [xs[0]]
        else:
            ys = F.array.split_axis.SplitAxis(
                len(self.lengths), axis=0).forward(xs)
        return tuple(y[0, :length] for y, length in zip(ys, self.lengths))

    def backward(self, indexes, grad_outputs):
        gy = F.pad_sequence(grad_outputs, self.inputs[0].shape[1])
        return gy,
