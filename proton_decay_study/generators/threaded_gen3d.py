# -*- coding: utf-8 -*-
from .base import BaseDataGenerator
from .single_file_3d import Gen3D
import logging
import threading
import queue
import traceback
import signal
import sys
import random


class SingleFileThread(threading.Thread):
  """
    Wrapper thread for buffering data from a 
    single file

    TODO: Looks like the destructor isn't killings
    the daughter threads in good time.
  """

  # Locks class for the duration of parent lifetime
  # Currently, hierarchal threading is not supported
  threadLock = threading.Lock()

  # Set this to 0 to kill all threads once prefetch is finished.
  __ThreadExitFlag__ = 1

  # Holds the current request queue
  queue = queue.Queue(10000)

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

    self._buffer = None
    self.single_thread_lock = threading.Lock()

    # This holds the file generator pattern
    self._filegen = None

  def run(self):
    """
      Loops over queue to accept new configurations
    """
    self.logger.info("Starting thread: {}".format(self))
    while SingleFileThread.__ThreadExitFlag__:
      # The thread loop should exit once it sees
      # the stop thread flag

      if self.single_thread_lock.locked() or self._buffer is not None:
        # If this is currently being visited by parent
        # Just continue
        continue
      # At this point, we need to fill the buffer

      # If we don't have  filegen, get one
      if self._filegen is None:
        self.queueLock.acquire()
        if not self.queue.empty():
          name = SingleFileThread.queue.get()
          try:
            self._filegen = Gen3D(name,
                                  self.datasetname,
                                  self.labelsetname,
                                  self.batch_size)
            self.logger.info("Moving to file: {}".format(self._filegen._filename))
          except Exception as e:
            self.logger.warning(e)
            self._filegen = None
            self.queueLock.release()
            return None
        else:
          self.queueLock.release()
          continue
        self.queueLock.release()
      # Now, fill the buffer and release
      self.single_thread_lock.acquire()
      try:
        self._buffer = self._filegen.next()
      except StopIteration:
        self._buffer = None
        self._filegen = None
      self.single_thread_lock.release()

  def visit(self, parent):
    # wait until we have it
    if self.single_thread_lock.locked():
      return None
    # Now grab it
    self.single_thread_lock.acquire()
    # Copy the data handle out (not the data itself)
    ret = self._buffer
    self._buffer = None
    self.single_thread_lock.release()
    parent.check_and_refill()
    return ret

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

  @staticmethod
  def status():
    SingleFileThread.logger.debug("ThreadLock: {}".format(SingleFileThread.threadLock.locked()))
    SingleFileThread.logger.debug("QueueLock: {}".format(SingleFileThread.queueLock.locked()))
    SingleFileThread.logger.debug("Flag: {}".format(SingleFileThread.__ThreadExitFlag__))

  def single_status(self):
    self.logger.debug("Single Thread Lock: {}".format(self.single_thread_lock.locked()))


# signal.signal(signal.SIGINT, SingleFileThread.killRunThreads)


class ThreadedMultiFileDataGenerator(BaseDataGenerator):
  """
    Uses threads to pull asynchronously from files
  """
  logger = logging.getLogger("pdk.gen.threaded_multi")

  def __init__(self, datapaths, datasetname,
               labelsetname, batch_size=1, nThreads=8):
    self.datapaths = [i for i in datapaths]
    self.nThreads = nThreads
    self.datasetname = datasetname
    self.labelsetname = labelsetname
    self.batch_size = batch_size

  def __enter__(self):

    SingleFileThread.threadLock.acquire()
    for i in range(len(datapaths)):
      random.shuffle(self.datapaths)

    self.check_and_refill()

    SingleFileThread.startThreads(self.nThreads, self.datasetname,
                                  self.labelsetname, self.batch_size)

    self.current_thread_index = 0
    self.logger.info("Threaded multi file generator ready for generation")

    return self

  def __exit__(self ,type, value, traceback):
    self.kill_child_processes()
    return True
    
  def kill_child_processes(self):
    self.status()
    SingleFileThread.__ThreadExitFlag__ = 0
    for t in SingleFileThread.activeThreads:
       t.join(10.0)
    SingleFileThread.activeThreads = []
    SingleFileThread.threadLock.release()

  def __del__(self):
    self.kill_child_processes()

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

  def status(self):
    self.logger.debug("Filenames: {}".format(self.datapaths))
    self.logger.debug("Active threads: {}".format(SingleFileThread.activeThreads))
    SingleFileThread.status()
    for i in SingleFileThread.activeThreads:
      i.single_status()

  def check_and_refill(self):
    SingleFileThread.queueLock.acquire()
    if SingleFileThread.queue.empty():
      for i in self.datapaths:
        SingleFileThread.queue.put(i)
    SingleFileThread.queueLock.release()

  def next(self):
    # see if there's any pre-fetched data

    # If the filename queue is empty, fill it back up again.
    # This ensures that files are all used up before they
    # are iterated over again.
    if self.current_thread_index >= len(SingleFileThread.activeThreads):
      self.current_thread_index = 0
    thread = SingleFileThread.activeThreads[self.current_thread_index]
    self.current_thread_index+=1
    ret = thread.visit(self)
    while ret is None:
      if self.current_thread_index == len(SingleFileThread.activeThreads):
        self.current_thread_index = 0
      thread = SingleFileThread.activeThreads[self.current_thread_index]
      self.current_thread_index+=1
      ret = thread.visit(self)
    return ret
