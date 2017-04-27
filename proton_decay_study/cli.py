# -*- coding: utf-8 -*-

import click
import logging
import sys
import logging
import tensorflow
from proton_decay_study.models.vgg16 import VGG16
from proton_decay_study.generators.multi_file import MultiFileDataGenerator

@click.command()
def main():
    logging.basicConfig(level=logging.INFO)

@click.command()
@click.argument('file_list', nargs=-1)
def standard_vgg_training(file_list):
  logging.basicConfig(level=logging.DEBUG)
  logger = logging.getLogger()
  sess = tf.Session()


  generator = MultiFileDataGenerator(file_list, 'image/wires','label/type', batch_size=1)
  model = VGG16(generator)
  training_output = model.fit_generator(generator, steps_per_epoch = 1000, 
                                      epochs=1000)
  model.save("trained_weights.h5")
  open('history.json','w').write(str(training_output))
  logger.info("Done.")

@click.command()
@click.argument('--steps', default=100, type=click.INT)
@click.argument('--epochs', default=1000, type=click.INT)
@click.argument('--weights',default=None, type=click.Path(exists=True))
@click.argument('--history', default='history.json')
@click.argument('--output',default='stage1.h5')
@click.argument('file_list', nargs=-1)
def advanced_vgg_training(steps, epochs,weights, history, output, file_list):
  logging.basicConfig(level=logging.DEBUG)
  logger = logging.getLogger()
  sess = tf.Session()

  generator = MultiFileDataGenerator(file_list, 'image/wires','label/type', batch_size=1)
  model = VGG16(generator)
  if weights is not None:
    model.load_weights(weights)
  training_output = model.fit_generator(generator, steps_per_epoch = steps, 
                                      epochs=epochs)
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

  generator = MultiFileDataGenerator(file_list, 'image/wires','label/type', batch_size=1)
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

if __name__ == "__main__":
    main()