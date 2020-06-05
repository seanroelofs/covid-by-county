from torch import nn
import torch


class NN(nn.Module):

    def __init__(self, hidden_layers):
        super(NN, self).__init__()
        self.layer_sizes = [79] + hidden_layers + [1]
        self.layers = nn.ModuleList()
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])) 

    def forward(self, x):
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if (i < len(self.layers) - 2):
                out = nn.BatchNorm1d(self.layer_sizes[i+1])(out)
                out = nn.ReLU()(out) 
                out = nn.Dropout(0.3)(out)
            if (i == len(self.layers) -2):
                out = nn.Tanh()(out)


        return out
        
