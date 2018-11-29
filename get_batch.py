
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
  def __init__(self, length, datadir):
      self.dataset = []
      self.make_dataset(length, datadir)
      self.GetCoordinate(datadir)
      self.createBoxes()



  def make_dataset(self, length, datadir):
      self.dataset = []
      # try:
      #     with open('bbox_data.pkl', 'rb') as f:
      #          unpickler= pickle.load(f)
      #          images = unpickler.load()
      # except EOFError:
      #     print(os.path.getsize('bbox_data.pkl'))

      cc = self.GetCoordinate(datadir)
      print(cc)
      images0 = np.zeros((7481, 512, 1392))
      images = self.createBoxes(images0, cc)
      num = images.shape[0]
      #pdb.set_trace()
      for i in range(num-length-2):
          for j in range(0,length-1,2):
              X = np.stack((images[i+j,:,:], images[i+j+1,:,:]), axis=0)
              #pdb.set_trace()
              Y = np.stack((images[i+j+1,:,:], images[i+j+2,:,:]), axis=0)
          self. dataset.append({"X": X, "Y": Y})
      return self.dataset

  def __len__(self):
      'Denotes the total number of samples'
      return len(self.dataset)

  def __getitem__(self, index):
       'Generates one sample of data'
       # Select sample

       return self.dataset[index]

  def save(self):
      pickle.dump(self, open(self.name + ".pkl", "wb"))

  def GetCoordinate(self,datadir):
      coordinate = []
      path = datadir + '*.txt'
      #path = './label_2' + '*.txt'
      files = glob.glob(path)
      for name in files:
          try:
              with open(name) as f:
                  cars = []
                  for i, t in enumerate(f.readlines(), 1):
                      l_s = t.split()
                      dic = {}
                      if l_s[0] == 'Car':
                          dic['x1'] = l_s[4]
                          dic['y1'] = l_s[5]
                          dic['x2'] = l_s[6]
                          dic['y2'] = l_s[7]
                          cars.append(dic)
              coordinate.append(cars)

          except IOError as exc:
              if exc.errno != errno.EISDIR:
                  print("error occurred here")
                  raise
      return coordinate

  def createBoxes(self, images, coordinates):
      seq_length = len(coordinates)
      for n in range(seq_length):
          frame = coordinates[n]
          if (n % 100 == 0):
              print("Processing the " + str(n) + "th image\n")
          for i in range(len(frame)):
              car = frame[i]
              pixel_start_x = int(round(float(car['x1'])))
              pixel_start_y = int(round(float(car['y1'])))
              pixel_end_x = int(round(float(car['x2'])))
              pixel_end_y = int(round(float(car['y2'])))
              for ri in range(pixel_end_y - pixel_start_y + 1):
                  for ci in range(pixel_end_x - pixel_start_x + 1):
                      images[n, pixel_start_y + ri, pixel_start_x + ci] = 1
      return images


def main():
    length = 10
    data_dir = './label_2/'
    dataset = AutorallyDataset(length, data_dir)
    dataset.save()


main()
