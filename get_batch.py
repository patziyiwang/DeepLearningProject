
from torch.utils.data import Dataset, DataLoader
import pickle
import errno
import glob
import numpy as np



class AutorallyDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, ):
      self.dataset = []
      self.make_dataset()



  def make_dataset(self):
      self.dataset = []
      length = 10
      ##we want to get images here img
      num = images.shape[2]
      for i in range(num-length-1):
          self. dataset.append({"x": images[:,:,i:i+length-1], "y": images[:,:,i+1, i+length]})

  def __len__(self):
      'Denotes the total number of samples'
      return len(self.dataset)

  def __getitem__(self, index):
       'Generates one sample of data'
       # Select sample
       return self.dataset[index]

  def save(self):
      pickle.dump(self, open(self.name + ".pkl", "wb"))




def main():
    dataset = AutorallyDataset()
    dataset.save()