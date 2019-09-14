import chainer
import chainer.functions as F


def _n_step_rnn_impl(
        f, n_layers, dropout_ratio, hx, cx, ws, bs, xs, use_bi_direction,
        recurrent_dropout_ratio, use_variational_dropout,
        apply_variational_dropout_to_input=False):
    direction = 2 if use_bi_direction else 1
    hx = F.separate(hx)
    use_cell = cx is not None
    if use_cell:
        cx = F.separate(cx)
    else:
        cx = [None] * len(hx)

    xs_next = xs
    hy = []
    cy = []
    # NOTE(chantera): Unlike Chainer, dropout is applied to inputs of the
    # first layer when enabling `apply_variational_dropout_to_input`.
    for layer in range(n_layers):
        # Forward RNN
        idx = direction * layer
        h_mask = None
        if layer == 0 and not apply_variational_dropout_to_input:
            xs = xs_next
        elif use_variational_dropout:
            x_mask = F.dropout(
                xs_next[0].data, recurrent_dropout_ratio, return_mask=True)[1]
            h_mask = F.dropout(
                hx[idx].data, recurrent_dropout_ratio, return_mask=True)[1]
            xs = _dropout_sequence(xs_next, dropout_ratio, x_mask)
        else:
            xs = _dropout_sequence(xs_next, dropout_ratio)
        h, c, h_forward = _one_directional_loop(
            f, xs, hx[idx], cx[idx], ws[idx], bs[idx],
            lambda h: F.dropout(h, recurrent_dropout_ratio, mask=h_mask))
        hy.append(h)
        cy.append(c)

        if use_bi_direction:
            # Backward RNN
            idx = direction * layer + 1
            h_mask = None
            if layer == 0 and not apply_variational_dropout_to_input:
                xs = xs_next
            elif use_variational_dropout:
                x_mask = F.dropout(
                    xs_next[0].data, recurrent_dropout_ratio,
                    return_mask=True)[1]
                h_mask = F.dropout(
                    hx[idx].data, recurrent_dropout_ratio, return_mask=True)[1]
                xs = _dropout_sequence(xs_next, dropout_ratio, x_mask)
            else:
                xs = _dropout_sequence(xs_next, dropout_ratio)
            h, c, h_backward = _one_directional_loop(
                f, reversed(xs), hx[idx], cx[idx], ws[idx], bs[idx],
                lambda h: F.dropout(
                    h, recurrent_dropout_ratio, mask=h_mask))
            h_backward.reverse()
            # Concat
            xs_next = [F.concat([hfi, hbi], axis=1)
                       for hfi, hbi in zip(h_forward, h_backward)]
            hy.append(h)
            cy.append(c)
        else:
            # Uni-directional RNN
            xs_next = h_forward

    ys = xs_next
    hy = F.stack(hy)
    if use_cell:
        cy = F.stack(cy)
    else:
        cy = None
    return hy, cy, tuple(ys)


def _one_directional_loop(f, xs, h, c, w, b, h_dropout):
    h_list = []
    for t, x in enumerate(xs):
        h = h_dropout(h)
        batch = len(x)
        need_split = len(h) > batch
        if need_split:
            h, h_rest = F.split_axis(h, [batch], axis=0)
            if c is not None:
                c, c_rest = F.split_axis(c, [batch], axis=0)

        h, c = f(x, h, c, w, b)
        h_list.append(h)

        if need_split:
            h = F.concat([h, h_rest], axis=0)
            if c is not None:
                c = F.concat([c, c_rest], axis=0)
    return h, c, h_list


def _dropout_sequence(xs, dropout_ratio, mask=None):
    if mask is not None:
        return [F.dropout(
            x, dropout_ratio, mask=mask[:x.shape[0]]) for x in xs]
    else:
        return [F.dropout(x, dropout_ratio) for x in xs]


def _n_step_lstm_base(
        n_layers, dropout_ratio, hx, cx, ws, bs, xs, use_bi_direction,
        recurrent_dropout_ratio=0.0, use_variational_dropout=False, **kwargs):
    if chainer.configuration.config.train and recurrent_dropout_ratio > 0.0:
        if use_variational_dropout:
            return _n_step_rnn_impl(
                F.connection.n_step_lstm._lstm,
                n_layers, dropout_ratio, hx, cx, ws, bs, xs, use_bi_direction,
                recurrent_dropout_ratio, use_variational_dropout)
        else:  # drop connect
            ws = [list(w) for w in ws]
            r_ws = []
            for i in range(len(ws)):
                r_ws.extend(ws[i][4:])
            r_ws = F.split_axis(F.dropout(
                F.concat(r_ws), recurrent_dropout_ratio), len(r_ws), axis=1)
            for i in range(len(ws)):
                ws[i][4:] = r_ws[4 * i: 4 * (i + 1)]
    return F.connection.n_step_lstm.n_step_lstm_base(
        n_layers, dropout_ratio, hx, cx, ws, bs, xs, use_bi_direction,
        **kwargs)


def n_step_lstm(
        n_layers, dropout_ratio, hx, cx, ws, bs, xs,
        recurrent_dropout_ratio=0.0, use_variational_dropout=False, **kwargs):
    return _n_step_lstm_base(
        n_layers, dropout_ratio, hx, cx, ws, bs, xs, False,
        recurrent_dropout_ratio, use_variational_dropout, **kwargs)


def n_step_bilstm(
        n_layers, dropout_ratio, hx, cx, ws, bs, xs,
        recurrent_dropout_ratio=0.0, use_variational_dropout=False, **kwargs):
    return _n_step_lstm_base(
        n_layers, dropout_ratio, hx, cx, ws, bs, xs, True,
        recurrent_dropout_ratio, use_variational_dropout, **kwargs)
