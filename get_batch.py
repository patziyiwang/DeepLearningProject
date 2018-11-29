
from torch.utils.data import Dataset, DataLoader
import pickle
import preprocess
from preprocess import *


class AutorallyDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        self.dataset = []
        self.make_dataset()



  def make_dataset(self):
      self.dataset = []
      ele_dict = {"x": }

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.dataset[index]

  def save(self):
      pickle.dump(self, open(self.name + ".pkl", "wb"))





dataset = AutorallyDataset(bags, name = "BetaDataset")
dataset.save()