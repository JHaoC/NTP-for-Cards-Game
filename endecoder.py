import torch as tr
import collections

# [[(((1, 0), 2), False, False), (((1, 1), 1), True, False), 
# (((1, 0), 2), False, False), (((0, 0), 2), True, False), 
# (((1, 1), 1), False, False), (((0, 1), 1), True, True)], 
# [(((0, 0), 2), False, False), (((1, 1), 1), True, False), 
# (((1, 0), 2), False, False), (((0, 0), 2), True, False), 
# (((1, 1), 1), False, False), (((0, 1), 1), True, True)]]
# convert game data to Matrix data for TSI and reverse Matrix 
# data to game data to trigger Game API

class Coder():
    def __init__(self, game_size):
        self.game_size = game_size
        self.options = game_size**2
        self.Mtable = []
        self.createMtable()

    def createMtable(self):
        for r in range(self.game_size):
            for c in range(self.game_size):
                self.Mtable.append((r,c))
        # self.Mtable.append(("checkGame"))
        # self.Mtable.append(("checkClick"))   

    def tranlatertensor(self,data):
        index_hop = tr.eye(self.options)
        res = []
        for br in data:
            times = []
            for env in br:
                indx_ = index_hop[env[0]]
                rest = tr.tensor(env[1:],dtype = tr.float32)
                detail = tr.cat((indx_,rest))
                times.append(detail)
            res.append(tr.stack(times))
        return res
                
    def encoder(self, data):
        if not data:
            return None
        self.batch = len(data)
        vals = []
        res_input = []
        res_output = []
        for bt in data:
            time_line_input = []
            time_line_output = []
            for env in bt:
                pos = self.Mtable.index(env[0][0])
                val = env[0][1]
                clickCheck_time = int(env[1][0])
                clickreward = env[1][1]
                gameCheck = int(env[2])
                time_line_input.append([pos,val,clickCheck_time,clickreward,gameCheck])
                time_line_output.append(pos)
            res_input.append(time_line_input)
            res_output.append(tr.tensor(time_line_output))
        return self.tranlatertensor(res_input), res_output

    def decoder(self, data):
        return self.Mtable[data]
        


if __name__ == "__main__":
    tranlater = Coder(2)
    data = [[(((0, 0), 1), (False, 0), False)]]
    # input_date = tranlater.encoder(data)
    a,b = tranlater.encoder(data)
    print(a)
    print(b)
    print(tranlater.decoder(0))
        

    
    