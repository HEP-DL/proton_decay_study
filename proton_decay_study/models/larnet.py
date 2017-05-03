import logging
import tensorflow as tf


class LArNet(object):
  logger = logging.getLogger('pdk.larnet')


  def __init__(self, generator, batch_size=1):
    self.generator = generator

    self.X = tf.placeholder(tf.float32, shape = [batch_size, generator.output])
    self.Y = tf.placeholder(tf.float32, shape = [batch_size, generator.input])

  def initialize(self):
    init = tf.global_variables_initializer()
    self.session = tf.Sess()
    self.session.run(init)

  def run(self):
    pass