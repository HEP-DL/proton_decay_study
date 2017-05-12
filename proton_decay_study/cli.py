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


@click.command()
def main():
    """
      For the moment, main just does main things
    """
    logging.basicConfig(level=logging.INFO)


@click.command()
@click.argument('file_list', nargs=-1)
def standard_vgg_training(file_list):
  """
    Standard VGG Training is aimed at 
  """
  logging.basicConfig(level=logging.DEBUG)
  logger = logging.getLogger()
  sess = tf.Session()


  generator = MultiFileDataGenerator(file_list, 'image/wires','label/type', batch_size=1)
  model = VGG16(generator)
  training_output = model.fit_generator(generator, steps_per_epoch = 1000, 
                                      epochs=1000)
  model.save("trained_weights.h5")
  open('history.json','w').write({'loss':training_output.history['loss'], 
                                  'accuracy':training_output['accuracy']})
  logger.info("Done.")

"""
_model = None
def signal_handler(signal, frame):
  global _model
  logging.info("SigINT was called. Saving Model")
  if _model is not None:
    _model.save('interrupted_output.h5')
    logging.info('Model Weights saved to interrupted_output.h5')
  sys.exit(0)
"""

@click.command()
@click.option('--steps', default=1000, type=click.INT)
@click.option('--epochs', default=1000, type=click.INT)
@click.option('--weights',default=None, type=click.Path(exists=True))
@click.option('--history', default='history.json')
@click.option('--output',default='stage1.h5')
@click.argument('file_list', nargs=-1)
def advanced_vgg_training(steps, epochs,weights, history, output, file_list):
  logging.basicConfig(level=logging.DEBUG)
  logger = logging.getLogger()
  sess = tf.Session()
  signal.signal(signal.SIGINT, signal_handler)
  from proton_decay_study.generators.threaded_multi_file import ThreadedMultiFileDataGenerator
  generator = ThreadedMultiFileDataGenerator(file_list, 'image/wires',
                                             'label/type', batch_size=1)
  model = VGG16(generator)
  global _model
  _model = model
  if weights is not None:
    model.load_weights(weights)
  logging.info("Starting Training")
  training_output = model.fit_generator(generator, steps_per_epoch = steps, 
                                      epochs=epochs, 
                                      workers=1,
                                      callbacks=[
                                        ModelCheckpoint(output, 
                                          monitor='val_loss', 
                                          verbose=0, 
                                          save_best_only=True, 
                                          save_weights_only=True, 
                                          mode='auto', 
                                          period=10)
                                      ])
  model.save(output)
  open(history,'w').write(str(training_output))
  logger.info("Done.")


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
                                      workers=1,
                                      verbose=1,
                                      max_q_size=8,
                                      pickle_safe=False,
                                      callbacks=[
                                        ModelCheckpoint(output, 
                                          monitor='val_loss', 
                                          verbose=0, 
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




if __name__ == "__main__":
    main()