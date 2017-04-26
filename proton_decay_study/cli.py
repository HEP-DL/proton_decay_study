# -*- coding: utf-8 -*-

import click
import logging
import sys
import logging

@click.command()
def main():
    logging.basicConfig(level=logging.INFO)

@click.command()
@click.argument('file_list', nargs=-1)
def standard_vgg_training(file_list):
  logging.basicConfig(level=logging.DEBUG)
  logger = logging.getLogger()
  from proton_decay_study.models.vgg16 import VGG16
  from proton_decay_study.generators.multi_file import MultiFileDataGenerator

  generator = MultiFileDataGenerator(file_list, 'image/wires','label/type', batch_size=1)
  model = VGG16(generator)
  training_output = model.fit_generator(generator, steps_per_epoch = 1000, 
                                      nb_epoch=1000)
  model.save("trained_weights.h5")
  open('history.json','w').write(training_output)
  logger.info("Done.")


if __name__ == "__main__":
    main()