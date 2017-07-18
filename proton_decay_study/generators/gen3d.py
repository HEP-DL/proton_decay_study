from proton_decay_study.generators.base import BaseDataGenerator
import h5py
import logging
import random
import numpy as np


class Gen3D(BaseDataGenerator):
  """
    Creates a generator for a list of files
  """
  logger = logging.getLogger("pdk.gen.gen3d")

  def __init__(self, datapaths, datasetname,
               labelsetname, batch_size=1):
    """
      Default generator.

      Args:
        datapaths: A list of filenames.
        datasetname: String representing dataset key in file for data.
        labelsetname: String representing dataset key in file for labels.
        batch_size: Integer of number of data frames to produce at once.
    """
    self._files = [i for i in datapaths]

    self._dataset = datasetname
    self._labelset = labelsetname
    self.batch_size = batch_size
    self.file_index = 0
    self.current_index = 0
    self.current_file = h5py.File(self._files[self.current_index], 'r')

  @property
  def output(self):
    """
      Output shape property

      Returns: A tuple representing the shape of the first data
      this picks out of the file
    """
    current_index = self.current_index
    file_index = self.file_index
    x, y = self.next()
    self.current_index = current_index
    self.file_index = file_index
    return x[0].shape

  @property
  def input(self):
    """
      Input shape property

      Returns: A tuple representing
    """
    current_index = self.current_index
    file_index = self.file_index
    x, y = self.next()
    self.current_index = current_index
    self.file_index = file_index
    return y[0].shape[0]

  def __len__(self):
    """
      Iterates over files to create the total sum length
      of the datasets in each file.
    """
    return sum([i[self._dataset].shape[0] for i in self._files])

  def next(self):

    if self.current_index >= self.current_file[self._dataset].shape[0]:
      # If we have to move to the next file in sequence
      next_file_index = self.file_index + 1
      if next_file_index >= len(self._files):
        next_file_index = 0
      msg = "Reached end of file: {} Moving to next file: {}"
      msg = msg.format(self._files[self.file_index],
                       self._files[next_file_index])
      self.logger.info(msg)
      self.file_index = next_file_index
      self.current_file = h5py.File(self._files[self.file_index], 'r')
      self.current_index = 0

    tmp_index = self.current_index + self.batch_size
    if tmp_index > self.current_file[self._dataset].shape[0]:
      # no longer allow stitching across files, remainder gets tossed
      self.current_index += tmp_index
      return self.next()
    # now we are guaranteed a file with enough spaces for a full batch

    tmp_x = self.current_file[self._dataset][self.current_index:tmp_index]
    x = np.ndarray(shape=(1, tmp_x.shape[0], tmp_x.shape[1],
                          tmp_x.shape[2], tmp_x.shape[3]))
    x[0] = tmp_x
    y = self.current_file[self._labelset][self.current_index:tmp_index]
    self.current_index += self.batch_size
    if len(x) == 0 or len(y) == 0 or not len(x) == len(y):
      return next(self)
    return (x, y)


class Gen3DRandom(Gen3D):
  def __init__(self, datapaths, datasetname,
               labelsetname, batch_size=1):
    super(Gen3DRandom, self).__init__(datapaths, datasetname,
                                      labelsetname, batch_size)
    for i in range(7):
      random.shuffle(self._files)
