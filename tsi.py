import torch as tr
import torch.nn as nn    

class TSI(nn.Module):

    def __init__(self, options, hidden_size, out_size, num_layers = 3):
        super(TSI, self).__init__()
        self.options = options
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = out_size
        self.lstm = nn.LSTM(options, hidden_size, num_layers, batch_first = True)
        self.out = nn.Linear(hidden_size,out_size)
        
    def forward(self, input_,hidden_layer_tuple = None):
        # sequence batch, size,Time step, layer size
        output, hidden_layer_tuple = self.lstm(input_, hidden_layer_tuple)
        output_= self.out(output)
        return output_, hidden_layer_tuple



