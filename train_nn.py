import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dtype = torch.Tensor


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=1):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)


    #Initialize hidden states h0, c0. Default is zero
    def init_hidden_zeros(self):
        self.h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        self.c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)

    def init_hidden(self, h0, c0):
        self.h0 = h0
        self.c0 = c0

    #Forward pass
    def forward(self, input):
        #Input to lstm has shape (seq_length, batch_size, input_size)
        #LSTM output has shape(seq_length, batch_size, hidden_size)
        #TODO: transform input into Rd
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1), (h0, c0))
        y_pred = self.linear(lstm_out)


def train(model, data, num_epochs, batch_size=32, lr=0.001, weight_decay=0.0, print_every=5):

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    loss_fn = nn.MSELoss  # Can experiment with different ones
    optimizer = optim.Adam(lr=lr, weight_decay=weight_decay)

    X = dtype(np.zeros((seq_length, batch_size, input_size)))
    Y = dtype(np.zeros((seq_length, batch_size, input_size)))

    for epoch in range(n_epochs):
        for batch_idx, batch in enumerate(dataloader):
            X = batch["input"]
            Y = batch["output"]
            optimizer.zero_grad()
            loss_fn(model(X), Y).backward()
            optimizer.step()



if __name__ == "__main__":

    #Initialize the parameters
    input_size = 32
    hidden_size = 64
    output_size = 32
    batch = 64
    n_layers = 2
    n_epochs = 100
    learning_rate = 0.0005
    weight_decay = 0.0

    #Create LSTM model
    model = LSTM(input_size, hidden_size, batch, output_dim=output_size, num_layers=n_layers)

    #Check if cuda is available
    if torch.cuda.is_available():
        model.cuda()

    train(model, data, n_epochs, batch_size=batch, lr=learning_rate, weight_decay=weight_decay)