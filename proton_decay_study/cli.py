# -*- coding: utf-8 -*-

import click
import logging
import sys
import logging
import tensorflow as tf
from proton_decay_study.models.vgg16 import VGG16
from proton_decay_study.generators.multi_file import MultiFileDataGenerator

from keras.callbacks import ModelCheckpoint, EarlyStopping
import signal
import sys
import os


@click.command()
@click.argument('n_gen', nargs=1, type=click.INT)
@click.argument('file_list', nargs=-1)
def test_file_input(n_gen, file_list):
  logging.basicConfig(level=logging.DEBUG)
  logger = logging.getLogger()
  from proton_decay_study.models.vgg16 import VGG16
  from proton_decay_study.generators.multi_file import MultiFileDataGenerator

  generator = MultiFileDataGenerator(file_list, 'image/wires',
                                     'label/type', batch_size=1)
  for i in range(int(n_gen)):
    x,y = generator.next()
    if len(x)==0:
      logging.warning("""Found NULL Frame
        File: {}
        Index: {}
        Batch Size: {}
        Remainder: {}
        Object: {}
        """.format(generator._files[generator.file_index],
                    generator.current_index,
                    generator.batch_size,
                    generator.current_index- len(generator._files[generator.file_index]),
                    (x,y)
          )
      )
  logger.info("Done.")

@click.command()
@click.argument('n_gen', nargs=1, type=click.INT)
@click.argument('file_list', nargs=-1)
def test_threaded_file_input(n_gen, file_list):
  logging.basicConfig(level=logging.DEBUG)
  logger = logging.getLogger()
  from proton_decay_study.models.vgg16 import VGG16
  from proton_decay_study.generators.multi_file import MultiFileDataGenerator

  generator = ThreadedMultiFileDataGenerator(file_list, 
                                            'image/wires','label/type', batch_size=1)
  logging.info("Now loading data...")
  for i in range(int(n_gen)):
    x,y = generator.next()
    if len(x)==0:
      logging.warning("""Found NULL Frame
        File: {}
        Index: {}
        Batch Size: {}
        Remainder: {}
        Object: {}
        """.format(generator._files[generator.file_index],
                    generator.current_index,
                    generator.batch_size,
                    generator.current_index- len(generator._files[generator.file_index]),
                    (x,y)
          )
      )
    else:
      logging.debug((x,y))
  logger.info("Done.")


@click.command()
@click.argument('input_file')
@click.argument('output_file', nargs=1)
def plot_model():
  from proton_decay_study.generators.multi_file import MultiFileDataGenerator
  gen = MultiFileDataGenerator(['../prod_pdk_nubarkplus_49.h5'],
                                'image/rawdigits','label/type')
  from proton_decay_study.models.vgg16 import VGG16
  from keras.utils.vis_utils import plot_model
  plot_model(model, show_shapes=True, to_file="vgg16.png")


@click.command()
@click.option('--steps', default=1000, type=click.INT)
@click.option('--epochs', default=1000, type=click.INT)
@click.option('--weights',default=None, type=click.Path(exists=True))
@click.option('--history', default='history.json')
@click.option('--output',default='stage1.h5')
@click.argument('file_list', nargs=-1)
def train_kevnet(steps, epochs,weights, history, output, file_list):
  from proton_decay_study.generators.gen3d import Gen3D
  from proton_decay_study.models.kevnet import Kevnet
  import tensorflow as tf
  logging.basicConfig(level=logging.DEBUG)
  logger = logging.getLogger()

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init)

  generator = Gen3D(file_list, 'image/wires','label/type', batch_size=1)
  model = Kevnet(generator)
  global _model
  _model = model
  if weights is not None:
    model.load_weights(weights)
  logging.info("Starting Training")
  training_output = model.fit_generator(generator, steps_per_epoch = steps, 
                                      epochs=epochs,
                                      workers=4,
                                      verbose=1,
                                      max_q_size=8,
                                      pickle_safe=False,
                                      callbacks=[
                                        ModelCheckpoint(output, 
                                          monitor='loss', 
                                          verbose=1, 
                                          save_best_only=True, 
                                          save_weights_only=True, 
                                          mode='auto', 
                                          period=10)
                                      ])
  model.save(output)
  training_history = {'epochs': training_output.epoch, 'acc': training_output.history['acc'], 'loss': training_output.history['loss']}
  import json
  open(history,'w').write(json.dumps(training_history))
  logger.info("Done.")

@click.command()
@click.option('--input', type=click.Path(exists=True))
@click.option('--weights', type=click.Path(exists=True))
def make_kevnet_featuremap(input, weights):
  from proton_decay_study.generators.gen3d import Gen3D
  from proton_decay_study.models.kevnet import Kevnet
  from proton_decay_study.visualization.kevnet import KevNetVisualizer
  logging.basicConfig(level=logging.DEBUG)
  logger = logging.getLogger()
  generator = Gen3D([input], 'image/wires','label/type', batch_size=1)
  model = Kevnet(generator)
  model.load_weights(weights)
  data = generator.next()
  vis = KevNetVisualizer(model, data)
  vis.initialize()
  vis.run()
  logger.info("Done.")


if __name__ == "__main__":
    main()