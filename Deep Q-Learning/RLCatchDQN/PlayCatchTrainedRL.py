from DQNetwork import DQNetwork
import torch
import numpy as np
from PIL import Image
from CatchGameEnv import CatchGameEnv
import matplotlib.pyplot as plt
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_frame_to2darray(images):
    x_t = images[0]
    img = Image.fromarray(x_t)
    x_t = img.resize(size=(100, 100))
    x_t = np.array(x_t).T
    x_t = x_t.astype('float')/np.max(x_t)
    # plt.imshow(x_t, cmap='gray')
    # plt.show()
    return x_t.reshape(1,100,100)

def main():
    model = DQNetwork().to(device)
    checkpoint_data = torch.load('checkpoint/modelcatch.pt')
    model.load_state_dict(checkpoint_data['state_dict'])
    num_wins, num_games = 0,0
    game = CatchGameEnv()
    for e in range(100):
        game.reset()
        a_0 = 1 # (0 = left, 1 = stay, 2 = right)
        game_over, x_t, r_t = game.mainGame(a_0)
        s_t = convert_frame_to2darray(x_t) # initial state
        while not game_over:
            q = model(torch.tensor(s_t).float().to(device)).cpu().detach()[0]
            action = np.argmax(q)
            game_over, x_t, r_t = game.mainGame(action)
            s_t = convert_frame_to2darray(x_t)
            if r_t == 1:
                num_wins += 1
        num_games +=1
        print("Games : %d | Wins : %d"%(num_games,num_wins))
if __name__ == "__main__":
    sys.exit(int(main() or 0))

