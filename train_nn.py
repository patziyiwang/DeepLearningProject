import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from get_batch import AutorallyDataset
from nn import LSTM
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pdb


def train_batch(model, dataset, n_epochs, seq_length=10, batch_size=32, lr=0.0001, weight_decay=0.0, grad_clip=0, print_every=5):

    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    end_idx = len(dataset)/batch_size

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(),lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        loss_total = 0
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx == end_idx:
                break
            input_seq = Variable(batch["input"].view(seq_length, batch_size, -1))
            output_seq = Variable(batch["output"].view(seq_length, batch_size, -1)) #Ground truth sequence
            if torch.cuda.is_available():
                input_seq = input_seq.cuda()
                output_seq = output_seq.cuda()
            optimizer.zero_grad()
            prediction, _ = model(input_seq)
            loss = loss_fn(prediction, output_seq)
            loss_total += loss.detach().item()
            loss.backward()
            if grad_clip != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if (epoch % print_every == 0):
            print("Training epoch: " + str(epoch) + "/" + str(n_epochs) + ", loss: " + str(loss_total))


def train_recurrent(model, dataset, n_epochs, seen_step, fut_step, batch_size=32, lr=0.00011, weight_decay=0.0, grad_clip=0, print_every=5):

    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    end_idx = len(dataset)/batch_size

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        loss_total = 0
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx == end_idx:
                break
            input_seq = Variable(batch["input"].view(seq_length, batch_size, -1))
            output_seq = Variable(batch["output"].view(seq_length, batch_size, -1))
            if torch.cuda.is_available():
                input_seq = input_seq.cuda()
                output_seq = output_seq.cuda()
            optimizer.zero_grad()
            _, (h, c) = model(input_seq[0:seen_step])
            empty_input = torch.zeros_like(input_seq[0:1])
            fut_prediction = []
            for t in range(fut_step):
                prediction, (h, c) = model.step(empty_input, h, c)
                fut_prediction.append(prediction)
            pred_seq = torch.cat(fut_prediction, dim=0)
            truth_seq = output_seq[seen_step:]
            loss = loss_fn(pred_seq, truth_seq)
            loss_total += loss.detach().item()
            loss.backward()
            if grad_clip != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if (epoch % print_every == 0):
            print("Training epoch: " + str(epoch) + "/" + str(n_epochs) + ", loss: " + str(loss_total))


def eval(model, dataset, seen_step, fut_step, batch_size=32):

    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    end_idx = len(dataset)/batch_size

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == end_idx:
            break
        input_seq = Variable(batch["input"].view(seq_length, batch_size, -1))
        output_seq = Variable(batch["output"].view(seq_length, batch_size, -1))
        if torch.cuda.is_available():
            input_seq = input_seq.cuda()
            output_seq = output_seq.cuda()
        _, (h, c) = model(input_seq[0:seen_step])
        empty_input = torch.zeros_like(input_seq[0:1])
        fut_prediction = []
        for t in range(fut_step):
            prediction, (h, c) = model.step(empty_input, h, c)
            fut_prediction.append(prediction)
        pred_seq = torch.cat(fut_prediction, dim=0)
        truth_seq = output_seq[seen_step:]


def loadData(data_path, file_name):
    return pickle.load(open(data_path + file_name + ".pkl", "r"))


def save_model(model_path, file_name, model):
    torch.save(model, model_path + file_name + ".pt")


def load_model(model_path, file_name):
    return torch.load(model_path + file_name + ".pt")


if __name__ == "__main__":
    #Tunable network&training parameters
    hidden_size = 16
    dropout = 0
    batch = 32
    n_layers = 1
    n_epochs = 100
    learning_rate = 1*10**-5
    weight_decay = 0.0
    grad_clip = 1

    #True if training recurrently(Given seen_step predict fut_step)
    recurrent = True
    seen_step = 5
    fut_step = 5
    seq_length = seen_step + fut_step

    #Load data
    data_path = ""
    data_file_name = "bbox_data"
    data = loadData(data_path, data_file_name)

    #Create/Load dataset from data
    # dataset = AutorallyDataset(seq_length, data)
    # dataset.save("testDataset")
    dataset = pickle.load(open("testDataset.pkl", "r"))
    print("Data loaded successfully\n")

    #Data dimensions
    input_size = data[0].shape[0]*data[0].shape[1]
    output_size = input_size

    #Model save path and name information
    model_save_path = ""
    model_name = "test"

    #Create LSTM model
    model = LSTM(input_size, hidden_size, batch, output_dim=output_size, num_layers=n_layers, dropout=dropout)
    model.double()

    #Move model to cuda if available
    if torch.cuda.is_available():
        model.cuda()
    print("Model built\n")

    if recurrent:
        train_recurrent(model, dataset, n_epochs, seen_step=seen_step, fut_step=fut_step, batch_size=batch, lr=learning_rate, weight_decay=weight_decay, grad_clip=grad_clip)
    else:
        train_batch(model, dataset, n_epochs, seq_length=seq_length, batch_size=batch, lr=learning_rate, weight_decay=weight_decay, grad_clip=grad_clip)

    save_model(model_save_path, model_name, model)