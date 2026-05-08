import sys
import os
from CatchGameEnv import CatchGameEnv
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
#from scipy.misc import imresize
from PIL import Image
from DQNetwork import DQNetwork
import cv2
import numpy as np
from PIL import Image
import torch
import torch.optim as optim

GAMMA = 0.99
INITIAL_EPSILON = 0.4
FINAL_EPSILON = 0.0001
MEMORY_SIZE = 50000
NUM_EPOCHS_OBSERVE = 50
NUM_EPOCHS_TRAIN = 8000 #8000
NUM_ACTIONS = 3
BATCH_SIZE = 32
NUM_EPOCHS = NUM_EPOCHS_OBSERVE+NUM_EPOCHS_TRAIN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, i, optim, fname):
    # ---------save the latest model---------
    print("----------saving model-----------------")
    checkpoint_data = {
    'epoch': i,
    'state_dict': model.state_dict(),
    'optimizer': optim.state_dict()
    }
    ckpt_path = os.path.join("checkpoint/" + fname)
    torch.save(checkpoint_data, ckpt_path)
    model.train()
def convert_frame_to2darray(images):
    x_t = images[0]
    img = Image.fromarray(x_t)
    x_t = img.resize(size=(100, 100))
    x_t = np.array(x_t).T
    x_t = x_t.astype('float')/np.max(x_t)
    # plt.imshow(x_t, cmap='gray')
    # plt.show()
    return x_t.reshape(1,100,100)

def get_next_batch(experience,model,num_actions,gamma,batch_size):
    batch_indices = np.random.randint(low=0,high=len(experience),size=batch_size)
    batch = [experience[i] for i in batch_indices]
    X = np.zeros((batch_size,1,100,100))
    Y = np.zeros((batch_size,num_actions))
    A = np.zeros((batch_size,1))
    for i in range(len(batch)):
        s_t, a_t, r_t, s_tp1, game_over = batch[i]
        X[i] = s_t
        A[i] = a_t
        Y[i] = model(torch.tensor(s_t).to(device).float()).cpu().detach()[0]
        Q_sa = np.max(model(torch.tensor(s_tp1).to(device).float()).cpu().detach()[0].numpy())
        if game_over:
            Y[i,a_t] = r_t
        else:
            Y[i,a_t] = r_t + gamma * Q_sa
    return X,Y,A

def main():
    # instantiate game and experience replay queue
    model = DQNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_criterion = torch.nn.MSELoss()

    game = CatchGameEnv()
    experience = collections.deque(maxlen=MEMORY_SIZE) # experience store
    num_games, num_wins = 0,0
    epsilon = INITIAL_EPSILON

    # loss and num_wins
    loss_list = []
    num_wins_list = []
    Q_values= []

    for e in range(NUM_EPOCHS):
        game.reset()
        loss = 0.0
        Qmax = 0
        # get first state
        a_0 = 1 # (0 = left, 1 = stay, 2 = right)
        game_over, x_t, r_t = game.mainGame(a_0)
        s_t = convert_frame_to2darray(x_t)
        a = 5
        while not game_over:
            model.eval()
            s_tm1 = s_t
            # next action
            if e<= NUM_EPOCHS_OBSERVE:
                a_t = np.random.randint(low=0,high=NUM_ACTIONS,size=1)[0]
            else:
                if np.random.rand()<=epsilon:
                    a_t = np.random.randint(low=0,high=NUM_ACTIONS,size=1)[0]
                else:
                    q = model(torch.tensor(s_t).to(device).float()).cpu().detach()[0]
                    Qmax = np.max(model(torch.tensor(s_t).to(device).float()).cpu().detach()[0].numpy())
                    a_t = np.argmax(q)

            # apply action, get reward
            game_over, x_t, r_t = game.mainGame(a_t)
            s_t = convert_frame_to2darray(x_t)
            # if reward, increment num_wins
            if r_t == 1: 
                num_wins += 1
            # store experience
            experience.append((s_tm1,a_t,r_t,s_t,game_over))
            if e>NUM_EPOCHS_OBSERVE:
                model.train()
                # finished observing now start training
                # get next batch
                X, Y, A = get_next_batch(experience,model,NUM_ACTIONS,GAMMA,BATCH_SIZE)
                optimizer.zero_grad()
                outputs = model(torch.tensor(X).to(device).float()) # forward pass
                outs = torch.zeros((X.shape[0],Y.shape[1])).to(device)
                for i in range(0,X.shape[0]):
                    outs[i,A[i]] = outputs[i,A[i]]
                loss_net = loss_criterion(outs,
                torch.tensor(Y).to(device).float())
                loss_net.backward()
                optimizer.step()
                loss += loss_net.item()
                #print('-------loss=',loss.item())

        # reduce epsilon gradually
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/NUM_EPOCHS
        print("Epoch : %d | Loss: %f | Win Count : %d | Qmax :%f"%(e,loss,num_wins,Qmax))
        loss_list.append(loss)
        num_wins_list.append(num_wins)
        Q_values.append(Qmax)
        if e%1000 == 0:
            save_model(model,e,optimizer,"modelcatch.pt")

    save_model(model,e,optimizer,"modelcatch.pt")
    
    #plot the graphs
    path = "C:/Users/Shariar Islam Saimon/Desktop/PHD in University of Bridgeport/1st Semester/CPEG591 - Reinforcement Learning/Assignment 5/RLCatchDQN_1147129/output"
    plt.plot(loss_list,label='loss')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(f"{path}/loss.png")
    plt.gcf().clear()
    
    plt.plot(Q_values,label='Qvalues')
    plt.title('Q vales over episodes')
    plt.xlabel('epoch')
    plt.ylabel('Q value')
    plt.savefig(f"{path}/qvalue.png")
    plt.gcf().clear()
    
    plt.plot(num_wins_list,label='Wins')
    plt.title('No of wins')
    plt.xlabel('epoch')
    plt.ylabel('wins')
    plt.savefig(f"{path}/wins.png")

if __name__ == "__main__":
    sys.exit(int(main() or 0))