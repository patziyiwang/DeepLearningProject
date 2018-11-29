import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import get_batch.py
from get_batch import AutorallyDataset
from torch.utils.data import Dataset, DataLoader

dtype = torch.Tensor


class LSTM(nn.Module):

    def __init__(self, input_dim, enc_dim, hidden_dim, batch_size, output_dim=1, num_layers=1, h0=None, c0=None):
        super(LSTM, self).__init__()
        self.input_dim = input_dim[0]*input_dim[1] #n_row*n_col
        self.enc_dim = enc_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.encoder = nn.Linear(self.input_dim, self.enc_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.output_dim)

        #Initialize hidden states, default is zero
        if h0 is None:
            self.h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
            self.c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        else:
            self.h0 = h0
            self.c0 = c0


    #Forward pass
    def forward(self, input):
        #Input to encoder has shape (seq_length, batch_size, n_row*n_col)
        enc_input = self.encoder(input.view(input.shape[0], self.batch_size, -1))
        #LSTM output has shape(seq_length, batch_size, hidden_dim)
        lstm_out, self.hidden = self.lstm(enc_input, (self.h0, self.c0))
        #Decoder output has shape (seq_length, batch_size, output_dim)
        y_pred = self.decoder(lstm_out)
        return y_pred



def train(model, data, num_epochs, batch_size=32, lr=0.001, weight_decay=0.0, print_every=5):
    length = 10
    data_dir = './label_2/'
    data = AutorallyDataset(length, data_dir)
    data.save()
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    loss_fn = nn.MSELoss  # Can experiment with different ones
    optimizer = optim.Adam(params=model.parameters(),lr=lr, weight_decay=weight_decay)

    X = dtype(np.zeros((seq_length, batch_size, input_size)))
    Y = dtype(np.zeros((seq_length, batch_size, input_size)))

    for epoch in range(n_epochs):
        loss_total = 0
        for batch_idx, batch in enumerate(dataloader):
            X = batch["input"]
            Y = batch["output"]
            optimizer.zero_grad()
            loss = loss_fn(model(X), Y)
            loss_total += loss.data
            loss.backward()
            optimizer.step()
        if (epoch % print_every == 0):
            print("Training epoch: " + str(epoch) + "/" + str(n_epochs) + ", loss: " + str(loss_total))


def loadData(data_path, file_name):
    return pickle.load(open(data_path + file_name + ".pkl", "r"))


def save_model(model_path, file_name, model):
    torch.save(model, model_path + file_name + ".pt")


def load_model(model_path, file_name):
    return torch.load(model_path + file_name + ".pt")


if __name__ == "__main__":

    #Import data
    data_path = ""
    data_file_name = ""
    data = loadData(data_path, data_file_name)

    #Model save path and name information
    model_save_path = ""
    model_name = ""

    #Input&Network parameters
    input_size = (512,1392)
    encoder_size = 256
    hidden_size = 64
    output_size = input_size[0]*input_size[1]

    #Tunable parameters
    batch = 64
    n_layers = 2
    n_epochs = 100
    learning_rate = 0.0005
    weight_decay = 0.0
    seq_length = 10

    #Create LSTM model
    model = LSTM(input_size, encoder_size, hidden_size, batch, output_dim=output_size, num_layers=n_layers)

    #Check if cuda is available
    if torch.cuda.is_available():
        model.cuda()

    train(model, data, n_epochs, batch_size=batch, lr=learning_rate, weight_decay=weight_decay)
    save_model(model_save_path, model_name, model)