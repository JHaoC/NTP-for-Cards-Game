from tsi import TSI
from endecoder import Coder
import torch as tr
import torch.nn as nn
import torch.optim as op
from torch.nn.functional import softmax
import matplotlib
matplotlib.interactive(True)
import matplotlib.pyplot as pt

class NTP(nn.Module):
    # Game size N demonstates the game create N*N matrix
    # options is #postions of matrix, in TSI which ala a output_s
    # input_size is N*N+Envs, Envs 1 for val, 1 for time of click, 1 for
    # result of click and 1 for condition of Game
    def __init__(self, game_size, hidden_size, num_layers = 3, lr = 0.2):
        super(NTP, self).__init__()
        self.game_size = game_size
        self.options = game_size*game_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = self.options+4
        self.lr = lr
        self.translater = Coder(game_size)
        self.controller = TSI(self.input_size, hidden_size, self.options, num_layers)
        self.hidden_layer_tuple = None

    def train(self, data, times = 20, plot_loss = True):
        inputdata, target = self.translater.encoder(data)
        loss_func = nn.CrossEntropyLoss()
        optimizer = op.Adam(self.controller.parameters(), self.lr)
        loss_data = []
        for step in range(times):
            for v, batch_data in enumerate(inputdata):
                out_lstm, _ = self.controller.forward(batch_data.unsqueeze(0)[:,:-1,:])
                target_batch = target[v].unsqueeze(0)
                loss = loss_func(out_lstm.reshape(-1,self.options),target_batch[:,1:].reshape(-1))
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                loss_data.append(loss.clone().detach())
        if plot_loss:
            pt.plot(loss_data)
            pt.show()
        return loss_data

    def predict(self, data):
        inputs, _ = self.translater.encoder(data)
        if not inputs:
            return None
        out_list = []
        for inpt in inputs:
            input_ = inpt.unsqueeze(0)
            output, self.hidden_layer_tuple = self.controller.forward(input_, self.hidden_layer_tuple)
            output = tr.argmax(softmax(output, dim=2),dim=2)
            out_list.append(self.translater.decoder(output))
        return out_list

if __name__ == "__main__":
    data =[[(((1, 0), 3), (False, 0), False), (((0, 0), 3), (True, 1), False), (((1, 1), 0), (False, 0), False), (((0, 1), 0), (True, 1), True)], [(((1, 0), 3), (False, 0), False), (((0, 0), 3), (True, 1), False), (((1, 1), 0), (False, 0), False), (((0, 1), 0), (True, 1), True)], [(((1, 0), 3), (False, 0), False), (((0, 0), 3), (True, 1), False), (((1, 1), 0), (False, 0), False), (((0, 1), 0), (True, 1), True)], [(((1, 0), 3), (False, 0), False), (((0, 0), 3), (True, 1), False), (((1, 1), 0), (False, 0), False), (((0, 1), 0), (True, 1), True)]]
    lrs = [0.1,0.2,0.5]
    datas = []
    for lr in lrs:
        model = NTP(2,20,lr = lr)
        data = model.train(data, plot_loss = False)
        print (lr)
        datas.append(data)
    pt.plot(datas[0])
    pt.plot(datas[1])
    pt.plot(datas[2])
    pt.show()
    # print(model.predict(test))