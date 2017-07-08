import chainer
from chainer import cuda
from chainer.utils import type_check
from chainer.functions.connection.convolution_2d import Convolution2DFunction

def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class GroupedConvolution2DFunction(chainer.Function):
  def __init__(self, stride=1, pad=0):
    super().__init__()
    self.conv = Convolution2DFunction(stride=stride, pad=pad)

  def check_type_forward(self, in_types):
    n_in = in_types.size()
    type_check.expect(2 <= n_in, n_in <= 3)

    x_type = in_types[0]
    w_type = in_types[1]
    type_check.expect(
      x_type.dtype.kind == 'f',
      w_type.dtype.kind == 'f',
      x_type.ndim == 4,
      w_type.ndim == 5,
      x_type.shape[1] == w_type.shape[0] * w_type.shape[2],
    )

    if type_check.eval(n_in) == 3:
      b_type = in_types[2]
      type_check.expect(
        b_type.dtype == x_type.dtype,
        b_type.ndim == 2,
        b_type.shape[0] == w_type.shape[0],
        b_type.shape[1] == w_type.shape[1],
      )

  def forward(self, inputs):
    xp = cuda.get_array_module(*inputs)
    x, W = inputs[:2]
    b = inputs[2] if len(inputs) == 3 else None
    y = xp.empty((W.shape[0], x.shape[0], W.shape[1], x.shape[2], x.shape[3]), dtype=x.dtype)

    c = W.shape[2]

    for i in range(len(W)):
      new_inputs = [x[:, i * c:(i + 1) * c, :], W[i]]

      if b is not None:
        new_inputs.append(b[i])

      y[i, :] = self.conv.forward(new_inputs)[0]

    y = xp.rollaxis(y, 1)
    y = y.reshape(y.shape[0], y.shape[1] * y.shape[2], *y.shape[3:])

    return y,

  def backward(self, inputs, grad_outputs):
    xp = cuda.get_array_module(*inputs)
    x, W = inputs[:2]
    b = inputs[2] if len(inputs) == 3 else None
    gy = grad_outputs[0]
    cx = W.shape[2]
    cy = W.shape[1]

    gx = xp.zeros((W.shape[0], x.shape[0], W.shape[2], *x.shape[2:]), dtype=x.dtype)
    gW = xp.zeros_like(W)
    gb = None

    if b is not None:
      gb = xp.zeros_like(b)

    for i in range(len(W)):
      new_inputs = [x[:, i * cx:(i + 1) * cx, :], W[i]]
      new_grad_outputs = [gy[:, i * cy:(i + 1) * cy, :]]

      if b is not None:
        new_inputs.append(b[i])

      g = self.conv.backward(new_inputs, new_grad_outputs)
      gx[i, :] = g[0]
      gW[i, :] = g[1]

      if gb is not None:
        gb[i, :] = g[2]

    gx = xp.rollaxis(gx, 1)
    gx = gx.reshape(gx.shape[0], gx.shape[1] * gx.shape[2], *gx.shape[3:])

    if gb is None:
      return gx, gW
    else:
      return gx, gW, gb


def grouped_convolution_2d(x, W, b=None, stride=1, pad=0):
  func = GroupedConvolution2DFunction(stride, pad)
  if b is None:
    return func(x, W)
  else:
    return func(x, W, b)

