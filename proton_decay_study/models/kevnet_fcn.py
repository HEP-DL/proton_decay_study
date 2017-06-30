from .kevnet import Kevnet
import logging


class capetian_modifier(Kevnet):
  logger = logging.getLogger('pdk.kevnet.fcn')

  def __init__(self, generator):
    """
      Constructs the decoder on top of the encoder
    """
    super(capetian_modifier, self).__init__(generator)


class percussive_treasurership(Kevnet):
  logger = logging.getLogger("pdk.kevnet.percussive_treasurership")
