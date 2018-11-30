from torch.utils.data import Dataset, DataLoader
import pickle
import errno
import glob
import numpy as np
import os
import errno
import glob
import numpy as np
import pickle
import pdb


class AutorallyDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, length, datainput):
      self.dataset = []
      self.make_dataset(length,datainput)

  def make_dataset(self, length, datainput):
      self.dataset = []
      # try:
      #     with open('bbox_data.pkl', 'rb') as f:
      #          unpickler= pickle.load(f)
      #          images = unpickler.load()
      # except EOFError:
      #     print(os.path.getsize('bbox_data.pkl'))

      # cc = self.GetCoordinate(datadir)
      # image_dim, scaling = self.getScaling(dim)
      # a = int(image_dim[0])
      # b = int(image_dim[1])
      # images0 = np.zeros((7481, a,b))
      print("Creating dataset\n")
      images = datainput
      num = images.shape[0]
      for i in range(num - length - 2):
          X = []
          Y = []
          for j in range(0, length - 1, 2):
              X.append(np.stack((images[i, :, :], images[i + j + 1, :, :]), axis=0))
              Y.append(np.stack((images[i + 1, :, :], images[i + j + 2, :, :]), axis=0))
          X = np.asarray(X)
          Y = np.asarray(Y)
          Xreal = np.zeros((length, X.shape[2], X.shape[3]))
          Yreal = np.zeros((length, X.shape[2], X.shape[3]))
          for k in range(X.shape[0]):
              m = k * 2
              Xreal[m, :, :] = X[k, 1, :, :]
              Xreal[m + 1, :, :] = X[k, 1, :, :]
              Yreal[m, :, :] = X[k, 1, :, :]
              Yreal[m + 1, :, :] = X[k, 1, :, :]

          self.dataset.append({"input": Xreal, "output": Yreal})
      return self.dataset

  def __len__(self):
      'Denotes the total number of samples'
      return len(self.dataset)

  def __getitem__(self, index):
       'Generates one sample of data'
       # Select sample

       return self.dataset[index]

  def save(self, data_name):
      pickle.dump(self, open(data_name + ".pkl", "wb"))

  # def GetCoordinate(self,datadir):
  #     coordinate = []
  #     path = datadir + '*.txt'
  #     #path = './label_2' + '*.txt'
  #     files = glob.glob(path)
  #     for name in files:
  #         try:
  #             with open(name) as f:
  #                 cars = []
  #                 for i, t in enumerate(f.readlines(), 1):
  #                     l_s = t.split()
  #                     dic = {}
  #                     if l_s[0] == 'Car':
  #                         dic['x1'] = l_s[4]
  #                         dic['y1'] = l_s[5]
  #                         dic['x2'] = l_s[6]
  #                         dic['y2'] = l_s[7]
  #                         cars.append(dic)
  #             coordinate.append(cars)
  #
  #         except IOError as exc:
  #             if exc.errno != errno.EISDIR:
  #                 print("error occurred here")
  #                 raise
  #     return coordinate

  # def createBoxes(images, coordinates, scaling):
  #     seq_length = len(coordinates)
  #     for n in range(seq_length):
  #         if (n % 100 == 0):
  #             print("Processing the " + str(n) + "th image\n")
  #         frame = coordinates[n]
  #         for i in range(len(frame)):
  #             car = frame[i]
  #             pixel_start_x = int(round(float(car['x1']) / scaling))
  #             pixel_start_y = int(round(float(car['y1']) / scaling))
  #             pixel_end_x = int(round(float(car['x2']) / scaling))
  #             pixel_end_y = int(round(float(car['y2']) / scaling))
  #             for ri in range(pixel_end_y - pixel_start_y + 1):
  #                 for ci in range(pixel_end_x - pixel_start_x + 1):
  #                     images[n, pixel_start_y + ri, pixel_start_x + ci] = 1
  #     print("Processing complete!\n")
  #     return images
  #
  # def getScaling(self, dim):
  #     scaling = 1
  #     while (dim[0] % 2 == 0 and dim[1] % 2 == 0 and min(dim) > 32):
  #         dim = (dim[0] / 2, dim[1] / 2)
  #         scaling *= 2
  #     return dim, scaling


def main():
    length = 10
    with open('bbox_data.pkl', 'rb') as f:
        datainput = pickle.load(f)
    dataset = AutorallyDataset(length,datainput)
    dataset.save("testDataset")
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    import pdb
    pdb.set_trace()

def test():
    with open('bbox_data.pkl', 'rb') as f:
        datainput = pickle.load(f)
    with open('datahere.pkl', 'rb') as f:
        dataset = pickle.load(f)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch_idx, batch in enumerate(dataloader):
        import pdb
        pdb.set_trace()
