from proton_decay_study.generators.base import BaseDataGenerator
import h5py
import logging


class MultiFileDataGenerator(BaseDataGenerator):
  """
    Creates a generator for a list of files
  """
  logger = logging.getLogger("pdk.gen.multi")

  def __init__(self, datapaths, datasetname,
               labelsetname, batch_size=10):
    self._files = [h5py.File(i, 'r') for i in datapaths]
    self._dataset = datasetname
    self._labelset = labelsetname
    self.batch_size = batch_size
    self.file_index = 0
    self.current_index = 0

  @property
  def output(self):
    current_index = self.current_index
    file_index = self.file_index
    x, y = self.next()
    self.current_index = current_index
    self.file_index = file_index
    return x[0].shape

  @property
  def input(self):
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
    """
      This should iterate over both files and datasets within a file.
    """
    tmp_index = self._files[self.file_index][self._dataset].shape[0]
    if self.current_index >= tmp_index:
      next_file_index = self.file_index + 1
      if next_file_index >= len(self._files):
        next_file_index = 0
      msg = "Reached end of file: {} Moving to next file: {}"
      msg.format(self._files[self.file_index], self._files[next_file_index])
      self.logger.info(msg)
      self.file_index += 1
    if self.file_index >= len(self._files):
      self.logger.info("Reached end of file stack. Now reusing data")
      self.file_index = 0
      self.current_index = 0
    tmp_index = self.current_index + self.batch_size
    if tmp_index > self._files[self.file_index][self._dataset].shape[0]:
      remainder = self._files[self.file_index][self._dataset].shape[0]
      remainder = abs(remainder - self.current_index)
      msg = "Crossing file boundary with remainder: {}".format(remainder)
      self.logger.info(msg)
      x = self._files[self.file_index][self._dataset][self.current_index:]
      y = self._files[self.file_index][self._labelset][self.current_index:]
      self.file_index += 1
      if self.file_index < len(self._files):
        msg = "Now moving to next file: {}"
        msg.format(self._files[self.file_index])
        self.logger.info(msg)
      self.current_index = remainder
      if len(x) == 0 or len(y) == 0 or not len(x) == len(y):
        return next(self)
      return (x, y)
    final_index = self.current_index + self.batch_size
    x = self._files[self.file_index]
    x = x[self._dataset][self.current_index:final_index]
    y = self._files[self.file_index]
    y = y[self._labelset][self.current_index:final_index]
    self.current_index += self.batch_size
    if len(x) == 0 or len(y) == 0 or not len(x) == len(y):
      return next(self)
    return (x, y)
