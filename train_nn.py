import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=1):
        super(LSTM, self).__init()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim), torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def __format__(self, input):
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        y_pred = self.linear(lstm_out)


def train(model, data, num_epochs):

if __name__ == "__main__":

    #Initialize the parameters
    input_size = 32
    hidden_size = 64
    output_size = 32
    batch = 32
    n_layers = 2
    n_epochs = 100
    learning_rate = 0.0005
    weight_decay = 0.0

    model = LSTM(input_size, hidden_size, batch, output_dim=output_size, num_layers=n_layers)

    #Check if cuda is available
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.MSELoss
    optimizer = optim.Adam(lr=learning_rate, weight_decay=weight_decay)

    train(model, data, n_epochs)