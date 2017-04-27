from proton_decay_study.generators.base import BaseDataGenerator
import h5py
import logging


class SingleFileDataGenerator(BaseDataGenerator):
  """
    Creates a generator for a single file
  """
  logger = logging.getLogger("pdk.gen.single")

  def __init__(self, datapath, dataset, labelset, batch_size=10):
    self._file = h5py.File(datapath,'r')
    self._dataset = self._file[dataset]
    self._labelset = self._file[labelset]
    self.current_index=0
    self.batch_size = batch_size

  def __len__(self):
    return self._dataset.shape[0]


  def next(self):
    if self.current_index>= len(self):
        self.logger.info("Reusing Data at Size: {}".format(len(self)))
        self.current_index = 0
    if self.current_index+self.batch_size>= len(self):
        self.batch_size = len(self)-current_index
    x = self._dataset[self.current_index:self.current_index+self.batch_size]
    y = self._labelset[self.current_index:self.current_index+self.batch_size]
    self.current_index+=self.batch_size
    return (x,y)