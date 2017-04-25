
class UBooNEDataGenerator(object):
  logger = logging.getLogger("uboone.data")
  def __init__(self, datapath, dataset, labelset):
    self.logger.info("Assembling DataSet")
    self._file = h5py.File(datapath,'r')
    self._dataset = self._file[dataset]
    self._labelset = self._file[labelset]
    self.current_index=0

  def __len__(self):
    return self._dataset.shape[0]

  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def next(self):
    batch_size = 10
    #This next bit causes the generator to loop indefinitely
    if self.current_index>= len(self):
        self.logger.info("Reusing Data at Size: {}".format(len(self)))
        self.current_index = 0
    if self.current_index+batch_size>= len(self):
        batch_size = len(self)-current_index
    x = self._dataset[self.current_index:self.current_index+batch_size]
    y = self._labelset[self.current_index:self.current_index+batch_size]
    self.current_index+=batch_size

    return (x,y)