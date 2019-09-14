import chainer.functions as F


def embed_id(x, W, ignore_label=None, fix_weight=False):
    return EmbedIDFunction(ignore_label, fix_weight).apply((x, W))[0]


class EmbedIDFunction(F.connection.embed_id.EmbedIDFunction):

    def __init__(self, ignore_label=None, fix_weight=False):
        super().__init__(ignore_label)
        self.fix_weight = fix_weight

    def backward(self, indexes, grad_outputs):
        if self.fix_weight:
            return None, None
        return super().backward(indexes, grad_outputs)
