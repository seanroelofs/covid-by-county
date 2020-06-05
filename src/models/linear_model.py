from torch import nn



class Linear(nn.Module):

    def __init__(self):
        super(Linear, self).__init__()
        self.fc1 = nn.Linear(in_features = 79, out_features = 1)

    def forward(self, x):
        out = self.fc1(x)
        return out
        
