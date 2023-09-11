import torch.nn as nn


class RNNFeatureAgent(nn.Module):
    """ Identical to rnn_agent, but does not compute value/probability for each action, only the hidden state. """
    def __init__(self, input_shape, args):
        nn.Module.__init__(self)
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)#45, 64
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):#[4,45]   [1,4,64]
        x = nn.functional.relu(self.fc1(inputs))#[4,64]
        h = self.rnn(x, hidden_state.reshape(-1, self.args.rnn_hidden_dim)) #[4,64]
        return None, h
