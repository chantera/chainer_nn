import chainer
import chainer.functions as F


def dropout(x, ratio=.5, **kwargs):
    """Disable dropout when ratio == 0.0."""
    if chainer.configuration.config.train and ratio > 0.0:
        return F.dropout(x, ratio, **kwargs)
    out, mask = chainer.as_variable(x), None
    if kwargs.get('return_mask', False):
        return out, mask
    return out
