# -*- coding: utf-8 -*-

import click
import logging
import sys
import logging
import tensorflow as tf
from proton_decay_study.models.vgg16 import VGG16
from proton_decay_study.generators.multi_file import MultiFileDataGenerator
from proton_decay_study.utils_kuza55.multi_gpu import make_parallel

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
import signal
import sys
import os


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
                                          period=1)
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
##@click.option('--input', default=None, type=click.Path(exists=True))
@click.option('--model_wts', default=None, type=click.Path(exists=True))
@click.argument('file_list', nargs=-1)
def plot_model(model_wts, file_list):
  from proton_decay_study.generators.multi_file import MultiFileDataGenerator
  from proton_decay_study.models.vgg16 import VGG16
  from keras.utils.vis_utils import plot_model
  from proton_decay_study.models.kevnet import Kevnet
  from proton_decay_study.generators.gen3d import Gen3D
  import pdb

##  from keras.models import load_model
## This falls over, not liking Kevnet.
##  model = load_model('./ectest.h5')
  
  generator = Gen3D(file_list, 'image/wires','label/type', batch_size=1)
  model = Kevnet(generator)
  model.load_weights(model_wts)

## This works after: pip install pyparsing==1.5.7; pip install pydot==1.0.28
#  plot_model(model, show_shapes=True, to_file='eckevnet.png')


### Below all fails, though, cuz sysadmin must install tk-inter before I can even pip install py-tkinter
#  import matplotlib.pyplot as plt
  import json
##  pdb.set_trace()

  with open("./jason.json") as json_file:
      jd = json.load(json_file)

#  plt(jd["epoch"],jd["acc"],label="insample accuracy")
#  plt(epoch,loss,label="insample loss")
#  plt.legend()
#  plt.plot()
#  plt.savefig('pi+_acc-loss.png')


@click.command()
@click.option('--steps', default=1000, type=click.INT)
@click.option('--epochs', default=1000, type=click.INT)
@click.option('--weights',default=None, type=click.Path(exists=True))
@click.option('--history', default='history.json')
@click.option('--output',default='stage1.h5')
@click.argument('file_list', nargs=-1)

def train_nbn(steps, epochs,weights, history, output, file_list):
  from proton_decay_study.generators.gen3d import Gen3D
  from proton_decay_study.models.nothinbutnet import Nothinbutnet
  import tensorflow as tf
  logging.basicConfig(level=logging.DEBUG)
  logger = logging.getLogger()

  init = tf.global_variables_initializer()

  with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)

#  import pdb
#  pdb.set_trace()  
  generator = Gen3D(file_list, 'image/wires','label/type', batch_size=1)

#  end = max(len(file_list)-10,0)
  import glob
#  file_list_v =  glob.glob("/microboone/ec/valid_singles/*")
  file_list_v =  glob.glob("/data/dlhep/quantized_h5files/*")
  validation_generator = Gen3D(file_list_v, 'image/wires', 'label/type', batch_size=1)

  model = Nothinbutnet(generator)
  global _model
  _model = model
  if weights is not None:
    model.load_weights(weights)
  logging.info("Starting Training")
  training_output = model.fit_generator(generator, steps_per_epoch = steps, 
                                      epochs=epochs,
#                                      workers=4,
                                      workers=1,
                                      verbose=1,
#                                      max_q_size=8,
                                      max_q_size=4,
                                      pickle_safe=False,
                                      validation_data=validation_generator,
                                      validation_steps = steps,
                                      callbacks=[
                                        ModelCheckpoint(output, 
                                          monitor='loss', 
                                          verbose=1, 
                                          save_best_only=True, 
                                          save_weights_only=True, 
                                          mode='auto', 
                                          period=10),
                                        ReduceLROnPlateau(monitor='loss', # I changed from val_loss
                                          factor=0.1, 
                                          patience=3, # epochs
                                          verbose=True, 
                                          mode='auto', 
                                          epsilon=0.0001, 
                                          cooldown=0, 
                                          min_lr=0.001)
#                                          TensorBoard(log_dir='./logs2',
#                                                      histogram_freq=1, 
#                                                      write_graph=True, 
#                                                      write_grads=True) 
#                                                      write_images=True)
                                      ])
  model.save(output)

#  pdb.set_trace()
#  pred10 = model.predict_generator(validation_generator,10) # 10 event predictions from last iteration of model -- I think
  import numpy as np
  pred10 = np.empty((0,10))
  label10 = np.empty((0,10))
#  for i in range(610): # (610):
  for i in range(10): # (610):
    x,y = validation_generator.next()
#    if i%100 == 0:
    logging.info( "getting a prediction and a label for event " + str(i))
    pred10 = np.row_stack((pred10,model.predict(x) ) ) # 10 event predictions from last iteration of model -- I think
    label10 = np.row_stack((label10,y) )  # 10 event predictions from last iteration of model -- I think


  training_history = {'epochs': training_output.epoch, 'acc': training_output.history['categorical_accuracy'], 'loss': training_output.history['loss'], 'val_acc': training_output.history['val_categorical_accuracy'], 'val_loss': training_output.history['val_loss'], 'val_predictions': np.around(pred10,decimals=3).tolist(), 'val_labels': np.around(label10,decimals=3).tolist() }
  import json
  open(history,'w').write(json.dumps(training_history))
  logger.info("Done.")


@click.command()
@click.option('--steps', default=1000, type=click.INT)
@click.option('--epochs', default=1000, type=click.INT)
@click.option('--weights',default=None, type=click.Path(exists=True))
@click.option('--history', default='history.json')
@click.option('--output',default='stage1.h5')
@click.argument('file_list', nargs=-1)
          
def train_nbn_prl(steps, epochs,weights, history, output, file_list):
  from proton_decay_study.generators.gen3d import Gen3D
  from proton_decay_study.models.nothinbutnet import Nothinbutnet
  import tensorflow as tf
  logging.basicConfig(level=logging.DEBUG)
  logger = logging.getLogger()

  init = tf.global_variables_initializer()

  with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)

  import pdb
  
  generator = Gen3D(file_list, 'image/wires','label/type', batch_size=1*4)
#  pdb.set_trace()
#  end = max(len(file_list)-10,0)
  import glob
  file_list_v =  glob.glob("/microboone/ec/valid_singles/*")
  validation_generator = Gen3D(file_list_v, 'image/wires', 'label/type', batch_size=1*4)

  model = Nothinbutnet(generator)
  global _model
  _model = model
  if weights is not None:
    model.load_weights(weights)
  logging.info("Starting Training")
  
  from utils_kuza55.multi_gpu import make_parallel
  model = make_parallel(model,4) # will use first 4 GPUs
  training_output = model.fit_generator(generator, steps_per_epoch = steps, 
                                      epochs=epochs,
#                                      workers=4,
                                      workers=1,
                                      verbose=1,
#                                      max_q_size=8,
                                      max_q_size=4,
                                      pickle_safe=False,
                                      validation_data=validation_generator,
                                      validation_steps = steps,
                                      callbacks=[
                                        ModelCheckpoint(output, 
                                          monitor='loss', 
                                          verbose=1, 
                                          save_best_only=True, 
                                          save_weights_only=True, 
                                          mode='auto', 
                                          period=10),
                                        ReduceLROnPlateau(monitor='loss', # I changed from val_loss
                                          factor=0.1, 
                                          patience=3, # epochs
                                          verbose=True, 
                                          mode='auto', 
                                          epsilon=0.0001, 
                                          cooldown=0, 
                                          min_lr=0.001),
                                          TensorBoard(log_dir='./logs2',
                                                      histogram_freq=1, 
                                                      write_graph=True, 
                                                      write_grads=True, 
                                                      write_images=True)
                                      ])

  model.save(output)

#  pdb.set_trace()
#  pred10 = model.predict_generator(validation_generator,10) # 10 event predictions from last iteration of model -- I think
  import numpy as np
  pred10 = np.empty((0,10))
  label10 = np.empty((0,10))
  for i in range(610):
      x,y = validation_generator.next()
      if i>590:
          pred10 = np.row_stack((pred10,model.predict(x) ) ) # 10 event predictions from last iteration of model -- I think
          label10 = np.row_stack((label10,y) )  # 10 event predictions from last iteration of model -- I think

          

  training_history = {'epochs': training_output.epoch, 'acc': training_output.history['acc'], 'loss': training_output.history['loss'], 'val_acc': training_output.history['val_acc'], 'val_loss': training_output.history['val_loss'], 'val_predictions': np.around(pred10.tolist(),decimals=3), 'val_labels': np.around(label10.tolist(),decimals=3) }
  import json
  open(history,'w').write(json.dumps(training_history))
  logger.info("Done.")


def train_nbn_prl(steps, epochs,weights, history, output, file_list):
  from proton_decay_study.generators.gen3d import Gen3D
  from proton_decay_study.models.nothinbutnet import Nothinbutnet
  import tensorflow as tf
  logging.basicConfig(level=logging.DEBUG)
  logger = logging.getLogger()

  init = tf.global_variables_initializer()

  with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)

  import pdb
  
  generator = Gen3D(file_list, 'image/wires','label/type', batch_size=1*4)
#  pdb.set_trace()
#  end = max(len(file_list)-10,0)
  import glob
  file_list_v =  glob.glob("/microboone/ec/valid_singles/*")
  validation_generator = Gen3D(file_list_v, 'image/wires', 'label/type', batch_size=1*4)

  model = Nothinbutnet(generator)
  global _model
  _model = model
  if weights is not None:
    model.load_weights(weights)

  model = make_parallel(model,4)
  logging.info("Starting Training")
  training_output = model.fit_generator(generator, steps_per_epoch = steps, 
                                      epochs=epochs,
#                                      workers=4,
                                      workers=1,
                                      verbose=1,
#                                      max_q_size=8,
                                      max_q_size=4,
                                      pickle_safe=False,
                                      validation_data=validation_generator,
                                      validation_steps = steps,
                                      callbacks=[
                                        ModelCheckpoint(output, 
                                          monitor='loss', 
                                          verbose=1, 
                                          save_best_only=True, 
                                          save_weights_only=True, 
                                          mode='auto', 
                                          period=10),
                                        ReduceLROnPlateau(monitor='loss', # I changed from val_loss
                                          factor=0.1, 
                                          patience=3, # epochs
                                          verbose=True, 
                                          mode='auto', 
                                          epsilon=0.0001, 
                                          cooldown=0, 
                                          min_lr=0.001),
                                          TensorBoard(log_dir='./logs',
                                                      histogram_freq=1, 
                                                      write_graph=True, 
                                                      write_grads=True, 
                                                      write_images=True)
                                      ])
  model.save(output)

#  pdb.set_trace()
#  pred10 = model.predict_generator(validation_generator,10) # 10 event predictions from last iteration of model -- I think
  import numpy as np
  pred10 = np.empty((0,10))
  label10 = np.empty((0,10))
  for i in range(610):
      x,y = validation_generator.next()
      if i>590:
          pred10 = np.row_stack((pred10,model.predict(x) ) ) # 10 event predictions from last iteration of model -- I think
          label10 = np.row_stack((label10,y) )  # 10 event predictions from last iteration of model -- I think

  training_history = {'epochs': training_output.epoch, 'acc': training_output.history['acc'], 'loss': training_output.history['loss'], 'val_acc': training_output.history['val_acc'], 'val_loss': training_output.history['val_loss'], 'val_predictions': np.around(pred10.tolist(),decimals=3), 'val_labels': np.around(label10.tolist(),decimals=3) }
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
