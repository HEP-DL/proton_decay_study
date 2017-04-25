import logging
import abc

class BaseDataGenerator(object)
  logger = logging.getLogger('pdk.generator')
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    self._dataset = None

  def __len__(self):
    return len(self._dataset)

  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  @abstractmethod
  def next(self):
    raise StopIteration()