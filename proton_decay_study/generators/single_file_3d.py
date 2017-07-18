from proton_decay_study.generators.base import BaseDataGenerator
import h5py
import logging
import random
import numpy as np


class Gen3D(BaseDataGenerator):
  """
    Creates a generator for a list of files
  """
  logger = logging.getLogger("pdk.gen.single.gen3d")

  def __init__(self, filename, datasetname,
               labelsetname, batch_size=1):
    """
      Default generator.

      Args:
        filename (str): Filename for this gen object
        datasetname: String representing dataset key in file for data.
        labelsetname: String representing dataset key in file for labels.
        batch_size: Integer of number of data frames to produce at once.
    """
    self._filename = filename
    self._dataset = datasetname
    self._labelset = labelsetname
    self.batch_size = batch_size
    self.current_index = 0
    self._file_len_ = None

  @property
  def output(self):
    """
      Output shape property

      Returns: A tuple representing the shape of the first data
      this picks out of the file
    """
    x, y = self.__get_next__()
    return x[0].shape

  @property
  def input(self):
    """
      Input shape property

      Returns: A tuple representing
    """
    x, y = self.__get_next__()
    return y[0].shape[0]

  def __len__(self):
    """
      Iterates over files to create the total sum length
      of the datasets in each file.
    """
    if self._file_len_ is None:
      _file = h5py.File(self._filename, 'r')
      self._file_len_ = _file[self._dataset].shape[0]
    return self._file_len_

  def __get_next__(self):
    end_index = self.current_index + self.batch_size
    if end_index >= len(self):
      raise StopIteration()

    _file = h5py.File(self._filename, 'r')
    tmp_x = _file[self._dataset][self.current_index:end_index]
    x = np.ndarray(shape=(1, tmp_x.shape[0], tmp_x.shape[1],
                          tmp_x.shape[2], tmp_x.shape[3]))
    x[0] = tmp_x
    y = np.array(_file[self._labelset][self.current_index:end_index], 
                 copy=True)
    if len(x) == 0 or len(y) == 0 or not len(x) == len(y):
      self.logger.warning("Encountered misshaped array")
      return next(self)
    return (x, y)

  def next(self):
      ret = self.__get_next__()
      self.current_index += self.batch_size
      return ret
