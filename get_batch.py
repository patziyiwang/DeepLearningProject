
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
  def __init__(self, ):
      self.dataset = []
      self.make_dataset()
      self.GetCoordinate()
      self.createBoxes()



  def make_dataset(self):
      self.dataset = []
      length = 10
      # try:
      #     with open('bbox_data.pkl', 'rb') as f:
      #          unpickler= pickle.load(f)
      #          images = unpickler.load()
      # except EOFError:
      #     print(os.path.getsize('bbox_data.pkl'))

      cc = self.GetCoordinate('./label_2')
      images0 = np.zeros((512, 1392, 7481))
      images = self.createBoxes(images0, cc)
      num = images.shape[2]
      #pdb.set_trace()
      for i in range(num-length-2):
          for j in range(0,length-1,2):
              X = np.concatenate((images[:,:,i+j], images[:,:,i+j+1]), axis=0)
              Y = np.concatenate((images[:,:,i+j+1], images[:,:,i+j+2]), axis=0)
          self. dataset.append({"x": X, "y": Y})

  def __len__(self):
      'Denotes the total number of samples'
      return len(self.dataset)

  def __getitem__(self, index):
       'Generates one sample of data'
       # Select sample
       return self.dataset[index]

  def save(self):
      pickle.dump(self, open(self.name + ".pkl", "wb"))

  def GetCoordinate(self, data_dir):
      coordinate = []
      path = data_dir + '*.txt'
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
                      images[pixel_start_y + ri, pixel_start_x + ci, n] = 1
      return images


def main():
    dataset = AutorallyDataset()
    dataset.save()


main()
