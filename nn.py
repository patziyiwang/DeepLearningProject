import torch
import torch.nn as nn


class LSTM(nn.Module):


    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=1, dropout=0, h0=None, c0=None):
        super(LSTM, self).__init__()
        self.input_dim = input_dim #n_row*n_col
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=dropout)
        self.decoder = nn.Linear(self.hidden_dim, self.output_dim)

        #Initialize hidden states, default is zero
        if h0 is None:
            self.h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
            self.c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        else:
            self.h0 = h0
            self.c0 = c0
        if torch.cuda.is_available():
            self.h0 = self.h0.cuda()
            self.c0 = self.c0.cuda()


    #Forward pass
    def forward(self, input):
        #Input to LSTM has shape (seq_length, batch_size, n_row*n_col)
        #LSTM output has shape (seq_length, batch_size, hidden_dim)
        lstm_out, self.hidden = self.lstm(input, (self.h0, self.c0))
        #Decoder output has shape (seq_length, batch_size, output_dim)
        prediction = self.decoder(lstm_out)
        return prediction, self.hidden

    #Propagate one step in forward pass
    def step(self, input, h, c):
        #Input to LSTM has shape (1, batch_size, n_row*n_col)
        #LSTM output has shape (1, batch_size, hidden_dim)
        if torch.cuda.is_available():
            h = h.cuda()
            c = c.cuda()
            input = input.cuda()
        lstm_out, self.hidden = self.lstm(input, (h, c))
        # Decoder output has shape (1, batch_size, output_dim)
        prediction = self.decoder(lstm_out)
        return prediction, self.hidden