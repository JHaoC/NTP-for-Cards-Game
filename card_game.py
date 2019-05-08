# A litte program to simulate a card game.
# There are bunches of cards faced down with a value on the table.
# Selete two cards and face them up, leave them if valus of them are same,
# other face them down again. Terminate game when face up all cards.
# Implement with python3 run it under python3
import functools
import tkinter as tk
import numpy as np
import time
import sys
import collections

def posiToInd(r,c,width):
        return r*width + c
    
def indToPost(ind,width):
        r = ind//width
        c = ind%width
        return r,c

class Card_Game(tk.Tk, object):
    def __init__(self, size=10,width =450, height = 300):
        super(Card_Game, self).__init__()
        self.title("Cards Game")
        self.positions = []
        self.size = size
        self.width = width
        self.height = height
        self.steps = 0
        self.hight_score = np.inf
        self.matric = None
        self.checked_elements = []
        self.geometry("{0}x{1}".format(width,height))
        self.check_list = []
        self.buttoms_list = []
        self.creatematric()
        # self.prefectSolution()
        self._build_card_game()


    def _build_card_game(self):
        self.buttoms = tk.Frame(self, width=6*self.size, height=3*self.size)
        self.others = tk.Frame(self, width=200, height=self.height)
        self.label_text_val= tk.StringVar()
        self.label_text = tk.Label(self.others, textvariable = self.label_text_val, width = 15, height = 2)
        self.label_text.pack()
        self.restart = tk.Button(self.others, text ="New Game", width = 20,
         height = 2, command = functools.partial(self.reset))
        self.restart.pack()
        
        self.label_hc_val = tk.StringVar()
        self.label_hc_val.set(str(self.hight_score))
        self.label_step_val= tk.StringVar()
        self.label_step_val.set(str(self.steps))
        self.label_step = tk.Label(self.others, text = "Total Steps", width = 15, height = 2)
        self.label_hc = tk.Label(self.others, text = "High Score", width = 15, height = 2)
        self.label_step_vr = tk.Label(self.others, textvariable = self.label_step_val, width = 30, height = 2)
        self.label_hc_vr = tk.Label(self.others, textvariable = self.label_hc_val, width = 30, height = 2)
        self.label_step.pack()
        self.label_step_vr.pack()
        self.label_hc.pack()
        self.label_hc_vr.pack()

        for r in range(self.size):
            tmp = []
            for c in range(self.size):
                self.positions.append((r,c))
                buttom = tk.Button(
                    self.buttoms, text="", width=3,
                    height=2, command=functools.partial(self.click, r, c),
                    anchor="center")
                buttom.grid(row=6*r, column=c*3)
                buttom.place()
                tmp.append(buttom)

            self.buttoms_list.append(tmp)

        self.buttoms.pack(side="left")
        self.others.pack(side = "right")
        # print(self.buttoms_list)

    def creatematric(self):
        try:
            half_size = self.size**2//2
            sample = np.random.randint(2*self.size, size=half_size)
            self.matric = np.append(sample, sample).reshape(
                self.size, self.size)
            np.random.shuffle(self.matric)
            print(self.matric)
        except Exception:
            print("Size must be Even")

    def faceUp(self, r, c):
        buttom = self.buttoms_list[r][c]
        val = self.matric[r, c]
        buttom.config(text=str(val))
        self.check_list.append((r, c))

    def faceDown(self, r, c):
        buttom = self.buttoms_list[r][c]
        buttom.config(text="")

    def check(self):
        if len(self.check_list) == 2:
            (r1,c1) = self.check_list.pop()
            (r2,c2) = self.check_list.pop()
            if self.matric[r1][c1] != self.matric[r2][c2]:
                self.after(500, self.faceDown,
                           r1, c1)
                self.after(500, self.faceDown,
                           r2, c2)
                # reward
                reward = -1
            else:
                #penish
                reward =1
                self.checked_elements.append((r1,c1))
                self.checked_elements.append((r2,c2))
            pos_ind1 = posiToInd(r1,c1,self.size)
            pos_ind2 = posiToInd(r2,c2,self.size)
            return [pos_ind1,pos_ind2,reward]
        else:
            return None

    def checkGameOver(self):
        if len(self.checked_elements) == self.size*self.size:
            return True
        return False

    def click(self, r, c):
        if (r,c) in self.checked_elements:
            return None
        if (r,c) in self.check_list:
            return None
        self.faceUp(r, c)
        res = self.check()
        self.steps += 1
        self.label_step_val.set(str(self.steps))
        if self.checkGameOver():
            self.label_text_val.set("Congratulation")
            self.hight_score = int(min(self.hight_score,self.steps))
            self.label_hc_val.set(str(self.hight_score))
            res[0] = "termial"
        return res

    def reset(self):
        self.steps = 0
        self.checked_elements = []
        self.label_step_val.set(str(self.steps))
        self.label_text_val.set("")
        # self.creatematric()
        for r in range(self.size):
            for c in range(self.size):
                self.faceDown(r,c)

    def prefectSolution(self):
        sol = []
        res = collections.defaultdict(list)
        for r in range(self.size):
            for c in range(self.size):
                v=self.matric[r,c]
                res[v].append((r,c))
                if len(res[v]) >1:
                    for _ in range(2):
                        posi = res[v].pop()
                        sol.append((posi,v))
        # print(sol)
        return sol
    
    def posiableSolution(self, successful_rate = 0.5):
        # increase time step
        sol = []
        checked = []
        res = collections.defaultdict(list)
        for r in range(self.size):
            for c in range(self.size):
                rate = (np.random.choice(10)/10 <= successful_rate)
                v=self.matric[r,c]
                res[v].append((r,c))
                # increase wrong case before success 
                if not rate:
                    rest_r = [i for i in range(r,self.size)]
                    rest_c = [i for i in range(c,self.size)]
                    for r_ in rest_r:
                        for c_ in rest_c:
                            v_ = self.matric[r_,c_]
                            if (r_,c_) in checked or v_ == v:
                                continue
                            else:
                                break
                    sol.append(((r,c),v))
                    sol.append(((r_,c_),v_))
                if len(res[v]) >1:
                    for _ in range(2):
                        posi = res[v].pop()
                        sol.append((posi,v))
                        checked.append(posi)

        # print(sol)
        return sol
    

if __name__ == '__main__':
    env = Card_Game(2)
    res = env.prefectSolution()
    res_p = env.posiableSolution(0.8)
    env.mainloop()
    

