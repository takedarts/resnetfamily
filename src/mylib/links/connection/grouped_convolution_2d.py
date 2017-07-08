from mylib.functions import grouped_convolution_2d
from chainer import initializers
from chainer import link
from chainer.utils import argument
from chainer import variable

def _pair(x):
  if hasattr(x, '__getitem__'):
    return x
  return x, x


class GroupedConvolution2D(link.Link):
  def __init__(self, in_channels, out_channels, units, ksize=None, stride=1, pad=0,
               nobias=False, initialW=None, initial_bias=None, **kwargs):
    super().__init__()

    argument.check_unexpected_kwargs(kwargs, deterministic="deterministic argument is not "
                                     "supported anymore. "
                                     "Use chainer.using_config('cudnn_deterministic', value) "
                                     "context where value is either `True` or `False`.")
    argument.assert_kwargs_empty(kwargs)

    if ksize is None:
      out_channels, ksize, in_channels = in_channels, out_channels, None

    self.units = units
    self.ksize = ksize
    self.stride = _pair(stride)
    self.pad = _pair(pad)
    self.out_channels = out_channels

    with self.init_scope():
      W_initializer = initializers._get_initializer(initialW)
      self.W = variable.Parameter(W_initializer)

      if in_channels is not None:
        self._initialize_params(in_channels)

      if nobias:
        self.b = None
      else:
        if initial_bias is None:
          initial_bias = 0

        bias_initializer = initializers._get_initializer(initial_bias)
        self.b = variable.Parameter(bias_initializer, (units, out_channels // units))

  def _initialize_params(self, in_channels):
    kh, kw = _pair(self.ksize)
    W_shape = (self.units, self.out_channels // self.units, in_channels // self.units, kh, kw)
    self.W.initialize(W_shape)

  def __call__(self, x):
    if self.W.data is None:
      self._initialize_params(x.shape[1])

    return grouped_convolution_2d(x, self.W, self.b, self.stride, self.pad)

