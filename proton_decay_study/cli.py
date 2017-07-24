# -*- coding: utf-8 -*-
import sys
import click
import logging
from proton_decay_study.generators.multi_file import MultiFileDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard


@click.command()
@click.argument('n_gen', nargs=1, type=click.INT)
@click.argument('file_list', nargs=-1)
def test_file_input(n_gen, file_list):
  logging.basicConfig(level=logging.DEBUG)
  logger = logging.getLogger()
  generator = MultiFileDataGenerator(file_list, 'image/wires',
                                     'label/type', batch_size=1)
  for i in range(int(n_gen)):
    x, y = generator.next()
    if len(x) == 0:
      bad_index = generator.current_index
      bad_index = bad_index - len(generator._files[generator.file_index])
      logging.warning("""Found NULL Frame
        File: {}
        Index: {}
        Batch Size: {}
        Remainder: {}
        Object: {}
        """.format(generator._files[generator.file_index],
                   generator.current_index,
                   generator.batch_size,
                   bad_index,
                   (x, y)
                   )
      )
  logger.info("Done.")


@click.command()
@click.argument('n_gen', nargs=1, type=click.INT)
@click.argument('file_list', nargs=-1)
def test_threaded_file_input(n_gen, file_list):
  logging.basicConfig(level=logging.DEBUG)
  logger = logging.getLogger()
  from .generators.threaded_multi_file import ThreadedMultiFileDataGenerator

  generator = ThreadedMultiFileDataGenerator(file_list,
                                             'image/wires',
                                             'label/type',
                                             batch_size=1)
  logging.info("Now loading data...")
  for i in range(int(n_gen)):
    x, y = generator.next()
    if len(x) == 0:
      bad_index = generator.current_index
      bad_index = bad_index - len(generator._files[generator.file_index])
      logging.warning("""Found NULL Frame
        File: {}
        Index: {}
        Batch Size: {}
        Remainder: {}
        Object: {}
        """.format(generator._files[generator.file_index],
                   generator.current_index,
                   generator.batch_size,
                   bad_index,
                   (x, y)
                   )
      )
    else:
      logging.debug((x, y))
  logger.info("Done.")


@click.command()
@click.option('--steps', default=1000, type=click.INT)
@click.option('--epochs', default=1000, type=click.INT)
@click.option('--weights', default=None, type=click.Path(exists=True))
@click.option('--history', default='history.json')
@click.option('--output', default='stage1.h5')
@click.argument('file_list', nargs=-1)
def train_kevnet(steps, epochs, weights, history, output, file_list):
  from proton_decay_study.generators.threaded_gen3d import ThreadedMultiFileDataGenerator
  from proton_decay_study.models.kevnet import Kevnet
  from proton_decay_study.callbacks.default import HistoryRecord
  logging.basicConfig(level=logging.DEBUG)
  logger = logging.getLogger()


  with ThreadedMultiFileDataGenerator(file_list, 'image/wires', 'label/type', batch_size=1) as generator:
    model = Kevnet(generator)
    global _model
    _model = model
    if weights is not None:
      model.load_weights(weights)
    model_checkpoint = ModelCheckpoint(output,
                                       monitor='loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='auto',
                                       period=10
                                       )
    history_checkpoint = CSVLogger(history.replace('.json','.csv'))
    logging.info("Starting Training")
    training_output = model.fit_generator(generator,
                                          use_multiprocessing=False,
                                          max_queue_size=1,
                                          verbose=1,
                                          workers=1,
                                          callbacks=[model_checkpoint,
                                                     history_checkpoint],
                                          epochs=epochs, 
                                          steps_per_epoch=steps)
    model.save(output)
    training_history = {'epochs': training_output.epoch,
                        'acc': training_output.history['acc'],
                        'loss': training_output.history['loss']}
    import json
    open(history, 'w').write(json.dumps(training_history))
    logger.info("Done.")


@click.command()
@click.option('--input', type=click.Path(exists=True))
@click.option('--weights', type=click.Path(exists=True))
def make_kevnet_featuremap(input, weights):
  from proton_decay_study.generators.threaded_gen3d import Gen3DRandom
  from proton_decay_study.models.kevnet import Kevnet
  from proton_decay_study.visualization.kevnet import KevNetVisualizer
  logging.basicConfig(level=logging.DEBUG)
  logger = logging.getLogger()
  generator = Gen3DRandom([input], 'image/wires', 'label/type', batch_size=1)
  model = Kevnet(generator)
  model.load_weights(weights)
  data = generator.next()
  vis = KevNetVisualizer(model, data)
  vis.initialize()
  vis.run()
  logger.info("Done.")


@click.command()
@click.option('--steps', default=1000, type=click.INT)
@click.option('--epochs', default=1000, type=click.INT)
@click.option('--weights', default=None, type=click.Path(exists=True))
@click.option('--history', default='history.json')
@click.option('--output', default='stage1.h5')
@click.argument('file_list', nargs=-1)
def train_widenet(steps, epochs, weights, history, output, file_list):
  from proton_decay_study.generators.threaded_gen3d import ThreadedMultiFileDataGenerator
  from proton_decay_study.models.widenet import Kevnet
  from proton_decay_study.callbacks.default import HistoryRecord
  logging.basicConfig(level=logging.DEBUG)
  logger = logging.getLogger()


  with ThreadedMultiFileDataGenerator(file_list, 'image/wires', 'label/type', batch_size=1) as generator:
    model = Kevnet(generator)
    global _model
    _model = model
    if weights is not None:
      model.load_weights(weights)
    model_checkpoint = ModelCheckpoint(output,
                                       monitor='loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='auto',
                                       period=1
                                       )
    history_checkpoint = HistoryRecord(history.replace('.json','.csv'))
    logging.info("Starting Training")
    training_output = model.fit_generator(generator,
                                          use_multiprocessing=False,
                                          max_queue_size=1,
                                          verbose=1,
                                          workers=1,
                                          callbacks=[model_checkpoint,
                                                     history_checkpoint],
                                          epochs=epochs, 
                                          steps_per_epoch=steps)
    model.save(output)
    training_history = {'epochs': training_output.epoch,
                        'acc': training_output.history['acc'],
                        'loss': training_output.history['loss']}
    import json
    open(history, 'w').write(json.dumps(training_history))
    generator.kill_child_processes()
    
    logger.info("Done.")


@click.command()
@click.option('--steps', default=1000, type=click.INT)
@click.option('--epochs', default=1000, type=click.INT)
@click.option('--weights', default=None, type=click.Path(exists=True))
@click.option('--history', default='history.json')
@click.option('--output', default='stage1.h5')
@click.option('--stage', default=0, type=click.INT)
@click.argument('file_list', nargs=-1)
def train_stagenet(steps, epochs, weights, history, output, stage, file_list):
  from proton_decay_study.generators.threaded_gen3d import ThreadedMultiFileDataGenerator
  from proton_decay_study.models.stages import stages
  from proton_decay_study.callbacks.default import HistoryRecord
  logging.basicConfig(level=logging.DEBUG)
  logger = logging.getLogger()

  with ThreadedMultiFileDataGenerator(file_list, 'image/wires', 'label/type', batch_size=1) as generator:
    model = stages[stage](generator)
    global _model
    _model = model
    if weights is not None:
      model.load_weights(weights, by_name=True)
    model_checkpoint = ModelCheckpoint(output,
                                       monitor='loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='auto',
                                       period=1
                                       )
    history_checkpoint = HistoryRecord(history.replace('.json','.csv'))
    es_callback =  EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=1, mode='auto')
    tb_callback = TensorBoard(write_grads=True, write_images=True, histogram_freq=10)
    logging.info("Starting Training [Moving to GPU]")
    training_output = model.fit_generator(generator,
                                          use_multiprocessing=False,
                                          max_queue_size=1,
                                          verbose=1,
                                          workers=1,
                                          callbacks=[model_checkpoint,
                                                     history_checkpoint, 
                                                     es_callback,
                                                     tb_callback],
                                          epochs=epochs, 
                                          steps_per_epoch=steps)
    model.save(output)
    training_history = {'epochs': training_output.epoch,
                        'acc': training_output.history['acc'],
                        'loss': training_output.history['loss']}
    import json
    with open(history, 'w') as history_output:
      history_output.write(json.dumps(training_history))
    logger.info("Done.")


@click.command()
@click.option('--output', default='.')
@click.option('--factor', default=6, type=click.INT)
@click.argument('file_list', nargs=-1)
def downsample_hep_files(output, factor, file_list):
  logging.basicConfig(level=logging.DEBUG)
  logger = logging.getLogger()
  import h5py
  for _filename in file_list:
    _file = h5py.File(_filename, 'r')