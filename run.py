import numpy as np
import time as tm
import matplotlib
matplotlib.use("TKAgg")
matplotlib.interactive(True)
import matplotlib.pyplot as pt
from card_game import Card_Game
from ntp import NTP

GAME_SIZE = 4
HIDDEN_SIZE = 50
NUM_LAYERS = 3
LR = 0.1

def checkGameOver_log():
    task_obs = env.checkGameOver()
    if task_obs:
        return True
    return False

def click_log(posi):
    observation = env.click(posi[0],posi[1])
    if observation:
        reward = observation[2]
        return True, reward
    return False, 0
 
def run_demo_loop(times, rate= 0.7, demo = False):
    for _ in range(times):
        log_once = []
        env.reset()
        solu = env.posiableSolution(rate)
        for step in solu:
            click_obs = click_log(step[0])
            task_obs = checkGameOver_log()
            log_once.append((step,click_obs,task_obs))
            if demo:
                env.update()
                tm.sleep(1)
        max_step.append(len(log_once))
        log.append(log_once)
        print("Demo_Sqes ", _, ": ", log_once)
    print ("In the demo, Max Time steps is ", max(max_step))

def run_game_NTP_mode(train_times, play_times, demo = True, plot_loss = True):
    complete_rates = []
    counts = []
    # Creating model and trainin using log as train data
    model.train(log, train_times, plot_loss)
    for _ in range(play_times):
        env.reset()
        # Clean hidden states of model
        if model.hidden_layer_tuple:
            model.hidden_layer_tuple = None
        # First step random chice observation
        r = np.random.randint(GAME_SIZE)
        c = np.random.randint(GAME_SIZE)
        posi, posi_old = (r,c), None
        task_obs = checkGameOver_log()
        count = 0
        while not task_obs:
            # fresh env
            if posi_old == posi:
                r = np.random.randint(GAME_SIZE)
                c = np.random.randint(GAME_SIZE)
                posi = (r,c)
            posi_old = posi
            count+=1
            print ("Times: ",count,", Click Positopm: ",posi)
            v = env.matric[posi[0],posi[1]]
            click_obs = click_log(posi)
            task_obs = checkGameOver_log()
            detail = [[((posi,v),click_obs,task_obs)]]
            posi = model.predict(detail)[0]
            if demo:
                env.update()
                tm.sleep(2)

            if count > 2*max(max_step):
                complete_rate = len(env.check_list)/float(GAME_SIZE**2)
                complete_rates.append(complete_rate)
                print ("Fail the game, already try double Max_demo steps, which is ", count)
                counts.append(count)
                break

        if count <= 2*max(max_step):
            complete_rates.append(1)
            counts.append(count)
            print ("Made it in ", count, "steps !!!")
    pt.figure(2)
    pt.plot(complete_rates)
    pt.figure(3)
    pt.plot(counts)
    pt.show()
    
        
def run(demo_seq_times, train_time, play_times, rate= 0.7, demo_loop = False, demo_NTP = True, plot_loss = True):
        run_demo_loop(demo_seq_times, rate, demo_loop)
        run_game_NTP_mode(train_time, play_times, demo_NTP, plot_loss)


# For general 
if __name__ == "__main__":
    log = []
    max_step = []
    env = Card_Game(GAME_SIZE)
    model = NTP(GAME_SIZE,HIDDEN_SIZE,NUM_LAYERS,LR)
    env.after(100, run, 4 , 80, 10, 0.8)
    env.mainloop()

    
