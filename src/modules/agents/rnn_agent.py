import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        # 所有 Agent 共享同一网络, 因此 input_shape = obs_shape + n_actions + n_agents（one_hot_code）=18+5+4
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim) #27--64
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim) #64---64
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)#64---5

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state): #[4,27],   [4,64]
        x = F.relu(self.fc1(inputs))  #[4,64]
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)  #[4,64]
        h = self.rnn(x, h_in)  # GRUCell 的输入要求（current_input, last_hidden_state）
        q = self.fc2(h)        # h 是这一时刻的隐状态，用于输到下一时刻的RNN网络中去，q 是真实行为Q值输出
        return q, h #q:[4,5]  h:[4,64]
