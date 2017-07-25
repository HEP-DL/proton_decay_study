


class DataSplit(object):
  def __init__(self, file_list):
    self.file_list = file_list

  def gen_train_validate(self):
    self.train = []
    self.test = []
    description={}

    for _file in self.file_list:
      head, tail = os.path.split(_file)
      group = tail.split("_")[:-1]
      if not group in description:
        description[group]=[]
      description[group].append(_file)

    for group in description:
      pass