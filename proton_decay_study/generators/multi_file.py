from proton_decay_study.generators.base import BaseDataGenerator
import h5py
import logging
import numpy as np

class MultiFileDataGenerator(BaseDataGenerator):
  """
    Creates a generator for a list of files
  """
  logger = logging.getLogger("pdk.gen.multi")

  def __init__(self, datapaths, datasetname, 
               labelsetname, batch_size=10, middle=False):
    self._files = [i for i in datapaths]
    import random
    random.shuffle(self._files)

    self._middle = middle

    self._dataset = datasetname
    self._labelset = labelsetname
    self.batch_size = batch_size
    self.file_index=0
    self.current_index=0
    self.truth = ["eminus", "eplus", "proton", "pizero", "piplus", "piminus", "muminus",  "muplus",  "kplus", "gamma"]
    self.labelvec = np.zeros(10)
    
    self.logger.info("Initializing h5 file object with value: {}".format(self._files[self.current_index]))

    self.current_file = h5py.File(self._files[self.current_index], 'r')
    self.handle1evtfiles(self.current_index)
    self.middlePane()

  def middlePane(self):
    # grab only some of the 3 MicroBooNE 3200-tick drifts. This works for MC. Data will be truncated, and we may want a different low, high here!
    if self._middle:
      low = 2200
      high = self.current_file['image'].shape[2] * 2/3
      high = 5400
      #      import pdb
#      pdb.set_trace()
#      print "clipping out only the middle time-ticks and wires " + str(low) +":" + str(high)
      x = self.current_file['image'][:,:,low:high,:]
      x[x<40] = 0
      self.current_file['image'] = x



  def handle1evtfiles(self,index):
    if len(self.current_file['image']) is not 2 and len(self.current_file['image'].shape) is 3: # These are Artem's single-event files.
      # We now proceed to make this data have the structure of the multi-event files.
      x = np.moveaxis(self.current_file['image'],2,0)
      y = np.expand_dims(x, axis=0)
      d = {}
#      pdb.set_trace()
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
#    return sum([i[self._dataset].shape[0] for i in self._files] )
    return len(self._files) * h5py.File(self._files[0],'r')[self._dataset].shape[0]

  """
  def next(self):
    if self.current_index >= self._files[self.file_index][self._dataset].shape[0]:
      next_file_index = self.file_index+1
      if next_file_index>= len(self._files):
        next_file_index=0
      self.logger.info("Reached end of file: {} Moving to next file: {}".format(self._files[self.file_index], 
                                                                                self._files[next_file_index]))
      self.file_index +=1
    if self.file_index >= len(self._files):
      self.logger.info("Reached end of file stack. Now reusing data")
      self.file_index = 0 
      self.current_index = 0
    if self.current_index+self.batch_size>self._files[self.file_index][self._dataset].shape[0]:

      remainder = abs(self._files[self.file_index][self._dataset].shape[0]- self.current_index)
      self.logger.info("Crossing file boundary with remainder: {}".format(remainder))
      x =  self._files[self.file_index][self._dataset][self.current_index:]
      y =  self._files[self.file_index][self._labelset][self.current_index:]




      self.file_index+=1
      if self.file_index<len(self._files):
        self.logger.info("Now moving to next file: {}".format(self._files[self.file_index]))
      self.current_index = remainder
      if len(x) == 0 or len(y)==0 or not len(x) == len(y):
        return next(self)
      return (x,y)

    x = self._files[self.file_index][self._dataset][self.current_index:self.current_index+self.batch_size]
    y = self._files[self.file_index][self._labelset][self.current_index:self.current_index+self.batch_size]
    self.current_index+=self.batch_size
    if len(x) == 0 or len(y)==0 or not len(x) == len(y):
      return next(self)
    return (x,y)
      """

## cutnpaste gen3d.py's next() here. EC, 21-Sep-2017
  def next(self):
    """
      This should iterate over both files and datasets within a file.
    """
    if self.current_index >= self.current_file[self._dataset].shape[0]:
      next_file_index = self.file_index+1
      if next_file_index>= len(self._files):
        next_file_index=0
      self.logger.info("Reached end of file: {} Moving to next file: {}".format(self._files[self.file_index], 
                                                                                self._files[next_file_index]))
      self.file_index +=1
      if self.file_index == len(self._files): self.file_index=0
      self.current_file = h5py.File(self._files[self.file_index], 'r')
      self.handle1evtfiles(self.file_index)
      self.middlePane()
      self.current_index=0
    if self.file_index >= len(self._files):
      self.logger.info("Reached end of file stack. Now reusing data")
      self.file_index = 0 
      self.current_index = 0


    multifile = False
    xapp = np.empty(np.append(1,self.current_file[self._dataset].shape))
    yapp = np.empty(self.current_file[self._labelset].shape)
    nevts = 0

    while self.batch_size > (self.current_file[self._dataset].shape[0]+nevts):
      multifile = True
      self.logger.info("Stitching together files")

      """
        This is the rare case of stitching together more than 1 file by crossing the boundary.
      """
      remainder = abs(self.current_file[self._dataset].shape[0]- self.current_index)
      self.logger.info("Crossing file boundary with remainder: {}".format(remainder))
      tmp_x =  self.current_file[self._dataset][self.current_index:]
      x = np.ndarray(shape=(1, tmp_x.shape[0],  tmp_x.shape[1],  tmp_x.shape[2],  tmp_x.shape[3]))
      x[0] = tmp_x 
      y = self.current_file[self._labelset][self.current_index:]
      self.file_index+=1
      if self.file_index == len(self._files): self.file_index=0
      self.current_file = h5py.File(self._files[self.file_index], 'r')
      self.handle1evtfiles(self.file_index)
      self.middlePane()
      self.current_index=0
      if self.file_index<len(self._files):
        self.logger.info("Now moving to next file: {}".format(self._files[self.file_index]))
      self.current_index = min(remainder,tmp_x.shape[0]-1)
      import pdb

      if len(x) == 0 or len(y)==0 or not len(x) == len(y):
        return next(self)
      xapp = np.append(xapp,x,axis=0)
      yapp = np.append(yapp,y,axis=0)
      nevts = xapp.shape[0]

    if multifile:
      # Send only plane 2.
      return (xapp[:,:,2,:,:],yapp)

#    pdb.set_trace()
    tmp_x = self.current_file[self._dataset][self.current_index:self.current_index+self.batch_size]
    x = np.ndarray(shape=(1, tmp_x.shape[0],  tmp_x.shape[1],  tmp_x.shape[2],  tmp_x.shape[3]))
    x[0] = tmp_x
    y = self.current_file[self._labelset][self.current_index:self.current_index+self.batch_size]
    self.current_index+=self.batch_size
    if len(x) == 0 or len(y)==0 or not len(x) == len(y):
      return next(self)
    # Send only plane 2.
    return (x[:,:,2,:,:],y)
      
