import torch
import torch.nn as nn

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        # Define the layers
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden state 初始隐状态
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device) #[1,1,20]

        # Forward pass through the RNN layer
        out, _ = self.rnn(x, h0) #[1,3,20]

        # Reshape the output to match the input of the fully connected layer
        out = out[:, -1, :]  #[1,20]

        # Forward pass through the fully connected layer
        out = self.fc(out)  #[1,5]

        return out

# Define the input size, hidden size, and output size
input_size = 10
hidden_size = 20
output_size = 5

# Create an instance of the RNN model
rnn = RNN(input_size, hidden_size, output_size)

# Generate a random input sequence
batch_size = 1
sequence_length = 3
input_sequence = torch.randn(batch_size, sequence_length, input_size) #[1,3,10]

# Forward pass through the RNN
output = rnn(input_sequence) #[1,5]

print("Output and its shape:", output, output.shape) 
# tensor([[ 0.0974,  0.2649,  0.0712, -0.1589,  0.0442]],   grad_fn=<AddmmBackward>) torch.Size([1, 5])