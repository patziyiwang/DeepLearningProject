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
