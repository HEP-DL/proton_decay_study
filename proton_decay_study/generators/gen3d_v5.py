from proton_decay_study.generators.base import BaseDataGenerator
import h5py
import logging
import numpy as np
import pdb
    
class Gen3D_v5(BaseDataGenerator):
  """
    Creates a generator for a list of files
  """
  logger = logging.getLogger("pdk.gen.gen3d")

  def __init__(self, datapaths, datasetname, 
               labelsetname, batch_size=10, middle=False):
    self._files = [ i for i in datapaths]
    import random
    random.shuffle(self._files)

    self._middle = middle
    self._dataset = datasetname
    self._labelset = labelsetname
    self.batch_size = batch_size
    self.file_index=0
    self.current_index=0
#    self.truth = ["eminus", "eplus", "proton", "pizero", "piplus", "piminus", "muminus",  "muplus",  "kplus", "gamma"]

#    self.truth = ["eplus", "eminus", "piminus", "muminus",   "kplus", "gamma"]
    self.truth = ["eminus", "piminus", "muminus",   "kplus"]
    self.labelvec = np.zeros(10)
    
    self.logger.info("Initializing h5 file object with value: {}".format(self._files[self.current_index]))


    self.file_index = int(np.random.randint(len(self._files), size=1))
    self.current_file = h5py.File(self._files[self.file_index], 'r')
    self.current_index = int(np.random.randint(self.current_file[self._dataset].shape[0], size=1))
    self.handlelabels(self.file_index)
    self.filterZeros()


  def filterZeros(self):
    # set low ADC pixes to 0
    pass

  def handlelabels(self,index):

    x = self.current_file['image']['wires']
    y = x
    d = {}

    ptype = str(self._files[index].split("/")[-1].split("_")[0]) # "kplus", say
    self._dataset = self.current_file.keys()[0]
    self._labelset = self.current_file.keys()[1]
    d[self._dataset] = y
    labelvectmp = np.array(self.labelvec)
#      print "handle1evtfiles: particle and index are: " + str(ptype) + "  " + str(self.truth.index(ptype))
    labelvectmp[self.truth.index(ptype)] = 1

    d[self._labelset] = np.expand_dims(labelvectmp,axis=0)
    self.current_file = d
    return

  @property
  def output(self):
    current_index= self.current_index
    file_index = self.file_index
    x,y = self.next()
    self.current_index =current_index
    self.file_index = file_index
    return x[0].shape

  @property
  def input(self):
    current_index= self.current_index
    file_index = self.file_index
    x,y = self.next()
    self.current_index =current_index
    self.file_index = file_index
    return y[0].shape[0]

  def __len__(self):
    """
      Iterates over files to create the total sum length
      of the datasets in each file.
    """
    ### uggh. This is v slow, especially with tons of 1-event files.
#    return sum([h5py.File(i,'r')[self._dataset].shape[0] for i in self._files] )
    ### Let's just take a guess. Not sure we use this number anyway.
#    return len(self._files) * h5py.File(self._files[0],'r')[self._dataset].shape[0] 
    return 12

  def debug_signal_handler(signal, frame):
    import pdb
    pdb.set_trace()
  import signal
#  signal.signal(signal.SIGINT, debug_signal_handler)

  def next(self):
    """
      This should iterate over both files and datasets within a file.
    """


#    import pdb

    multifile = False

    xapp = np.empty(np.append(1,np.append(1,self.current_file[self._dataset].shape[1:])))
    yapp = np.empty(self.current_file[self._labelset].shape)
    nevts = 0

    while self.batch_size > nevts:
      self.file_index = int(np.random.randint(len(self._files), size=1))
      self.current_file = h5py.File(self._files[self.file_index], 'r')
      self.handlelabels(self.file_index)
      self.filterZeros()
      self.current_index = int(np.random.randint(self.current_file[self._dataset].shape[0], size=1))

      self.logger.info("Reading h5 file: {}".format(self._files[self.file_index]))
      multifile = True

      tmp_x =  self.current_file[self._dataset][self.current_index] # Note, no longer ":"! Just take one event. EC, 4-Oct-2017.
      x = np.ndarray(shape=(1, 1, tmp_x.shape[0],  tmp_x.shape[1],  tmp_x.shape[2]))
      x[0] = tmp_x 
      y = self.current_file[self._labelset]


      if len(x) == 0 or len(y)==0 or not len(x) == len(y):
        return next(self)

      xapp = np.append(xapp,x,axis=0)
      yapp = np.append(yapp,y,axis=0)
      nevts += xapp.shape[0]

    if multifile:
      return (xapp,yapp)

    tmp_x = self.current_file[self._dataset][self.current_index:self.current_index+self.batch_size]
    x = np.ndarray(shape=(1, tmp_x.shape[0],  tmp_x.shape[1],  tmp_x.shape[2]))
    x[0] = tmp_x
    y = self.current_file[self._labelset]

    if len(x) == 0 or len(y)==0 or not len(x) == len(y):
      return next(self)
    return (x,y)
