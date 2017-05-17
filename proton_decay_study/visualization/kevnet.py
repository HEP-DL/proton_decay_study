from proton_decay_study.visualization.intermediate import IntermediateVisualizer
import logging
import os
import numpy

class KevNetVisualizer:
  layers = ['block1_conv1',
            'block2_conv1',
            'block3_conv1',
            'block4_conv1',
            'block5_conv1',
  ]
  logger = logging.getLogger("pdk.vis.kevnet")
  def __init__(self, model, data):
    self.model = model
    self.data = data

  def initialize(self):
    self.mkdir()
    output_path = os.path.join("featuremaps", 'data'+".npy")
    numpy.save(output_path, self.data[0])

  def mkdir(self):
    if not os.path.isdir("featuremaps"):
      self.logger.info("Creating feature maps directory")
      os.mkdir("featuremaps")

  def run(self):
    for layer in self.layers:
      self.logger.info("analyzing layer: "+layer)
      vis = IntermediateVisualizer(self.model, layer, self.data[0])
      output = vis.infer()
      output_path = os.path.join("featuremaps", layer+".npy")
      numpy.save(output_path, output)
