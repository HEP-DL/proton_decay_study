from proton_decay_study.generators.base import BaseDataGenerator
from proton_decay_study.generators.gen3d import SingleFileDataGenerator
import logging
import threading
import Queue
import traceback
import signal
import sys


class SingleFileThread(threading.Thread):
  """
    Represents a single file that is asynchronously
  """

  # Locks class for the duration of parent lifetime
  # Currently, hierarchal threading is not supported
  threadLock = threading.Lock()

  # Set this to 0 to kill all threads once prefetch is finished.
  __ThreadExitFlag__ = 1

  # Holds the current thread queue
  queue = Queue.Queue(10000)

  # Locks the thread queue
  queueLock = threading.Lock()
  logger = logging.getLogger('pdk.gen.singlefilethread')

  activeThreads = []

  def __init__(self, datasetname,
               labelsetname, batch_size):
    """
        :param: datasetname The name used to pull the dataset from file
        :type: datasetname string
        :param: labelsetname The name used to pull the labelset from file
        :type: labelsetname string
        :param: batch_size The number of images and truths to pull
        :type: batch_size int
    """
    super(SingleFileThread, self).__init__()
    self.datasetname = datasetname
    self.labelsetname = labelsetname
    self.batch_size = batch_size

    # prefetched means that there's data ready
    self._buffer = None
    self.single_thread_lock = threading.Lock()

    # This holds the
    self._filegen = None

  @property
  def get(self):
    self.single_thread_lock.acquire()
    if self._buffer is None:
      pass

  def run(self):
    """
      Loops over queue to accept new configurations
    """
    while SingleFileThread.__ThreadExitFlag__:
      if self.single_thread_lock.locked():
        continue
      else:
        self.single_thread_lock.acquire()
        if self._buffer is not None:
          self.single_thread_lock.release()
          continue
        self.single_thread_lock.release()

      if self._filegen is None or self._filegen.reused:
        self.queueLock.acquire()
        if not self.queue.empty():

          self._filegen = SingleFileDataGenerator(SingleFileThread.queue.get(),
                                                  self.datasetname,
                                                  self.labelsetname,
                                                  self.batch_size)
          self.logger.info("Moving to file: {}".format(self._filegen))
        self.queueLock.release()
      else:
        try:
          try:
            self.single_thread_lock.acquire()
            self._buffer = self._filegen.next()
          except StopIteration:
            self.logger.warning("Hit Stop Iteration")
            self._buffer = None
            self._filegen = None
          self.single_thread_lock.release()
        except Exception:
          exc_type, exc_value, exc_traceback = sys.exc_info()
          self.logger.error(repr(traceback.format_exception(exc_type,
                                                            exc_value,
                            exc_traceback)))

  @staticmethod
  def killRunThreads(signum, frame):
      """
          Sets the thread kill flag to each of the ongoing analysis threads
      """
      SingleFileThread.logger.info("Killing Single File threads...")
      SingleFileThread.__ThreadExitFlag__ = 0
      sys.exit(signum)

  @staticmethod
  def startThreads(nThreads, datasetname,
                   labelsetname, batch_size):
    msg = "Starting {} Single File threads".format(nThreads)
    SingleFileThread.logger.info(msg)
    for i in range(nThreads):
        thread = SingleFileThread(datasetname,
                                  labelsetname, batch_size)
        thread.start()
        SingleFileThread.activeThreads.append(thread)
    SingleFileThread.logger.info("Threads successfully started")

  @staticmethod
  def waitTillComplete(callback=None):
      queue = SingleFileThread
      if callback is None:
          while not queue.queue.empty() and queue.__ThreadExitFlag__:
              sys.stdout.flush()
      else:
          while not queue.queue.empty() and queue.__ThreadExitFlag__:
              callback()

      # Notify threads it's time to exit
      SingleFileThread.__ThreadExitFlag__ = 0

      # Wait for all threads to complete
      for t in SingleFileThread.activeThreads:
          t.join()
      # dealloc
      SingleFileThread.activeThreads = []


signal.signal(signal.SIGINT, SingleFileThread.killRunThreads)


class ThreadedMultiFileDataGenerator(BaseDataGenerator):
  """
      Uses threads to pull asynchronously from files
  """
  logger = logging.getLogger("pdk.gen.threaded_multi")

  def __init__(self, datapaths, datasetname,
               labelsetname, batch_size=1, nThreads=2):

    SingleFileThread.threadLock.acquire()
    self._threads = SingleFileThread.startThreads(nThreads, datasetname,
                                                  labelsetname, batch_size)
    SingleFileThread.queueLock.acquire()
    for config in datapaths:
        SingleFileThread.queue.put(config)
    SingleFileThread.queueLock.release()

    self.datapaths = datapaths
    self.logger.info("Threaded multi file generator ready for generation")
    self.current_thread_index = 0

  def __del__(self):
    SingleFileThread.__ThreadExitFlag__ = 0
    for t in SingleFileThread.activeThreads:
      t.join()
    del self._threads
    SingleFileThread.threadLock.release()

  @property
  def output(self):
    x, y = self.next()
    return x[0].shape

  @property
  def input(self):
    x, y = self.next()
    return y[0].shape[0]

  def __len__(self):
    """
      Iterates over files to create the total sum length
      of the datasets in each file.
    """
    return 0

  def next(self):
    # see if there's any pre-fetched data

    # If the filename queue is empty, fill it back up again.
    # This ensures that files are all used up before they
    # are iterated over again.
    self.logger.debug("Getting next")
    SingleFileThread.queueLock.acquire()
    if SingleFileThread.queue.empty():
      for i in self.datapaths:
        SingleFileThread.queue.put(i)
    SingleFileThread.queueLock.release()

    tmp_buffer = None
    while tmp_buffer is None:
      if self.current_thread_index >= len(SingleFileThread.activeThreads):
        self.current_thread_index = 0
      thread = SingleFileThread.activeThreads[self.current_thread_index]
      thread.single_thread_lock.acquire()
      if thread._buffer is not None:
        new_buff = thread._buffer
        thread._buffer = None
        thread.single_thread_lock.release()
        if new_buff is None:
          self.logger.warning("Found null dataset")
          return self.next()
        x, y = new_buff
        if not len(x) == len(y):
          self.logger.warning("Found incorrect dataset size")
          return self.next()
        return new_buff
      else:
        self.current_thread_index += 1
      thread.single_thread_lock.release()
    self.logger.warning("Threaded buffer found queue is empty. Trying again")
    return self.next()
