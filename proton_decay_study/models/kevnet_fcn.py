from .kevnet import Kevnet
import logging


class Kevnet_FCN(Kevnet):
  logger = logging.getLogger('pdk.kevnet.fcn')

  def __init__(self, generator):
    """
      Constructs the decoder on top of the encoder
    """
    super(Kevnet_FCN, self).__init__(generator)
