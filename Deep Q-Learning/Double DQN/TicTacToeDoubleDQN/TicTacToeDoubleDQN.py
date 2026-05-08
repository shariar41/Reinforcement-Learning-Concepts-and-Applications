import os
import time
import torch
import torch.nn.functional as F
import sys
from train import train
from TicTacToe import TicTacToe
from Network import Network
import numpy as np


def play(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TicTacToe()
    
    done = False
    obs, _ = env.reset()
    exp = {}
    player = 0

    while not done:
        time.sleep(1)
        print(
            "Commands:\n{}|{}|{}\n-----\n{}|{}|{}\n-----\n{}|{}|{}\n\nBoard:".format(
                *[x for x in range(0, 9)]
            )
        )
        env.render()  # display current state
        action = None

        # human player (player 1) is displayed as state 2 in the cell
        # computer (player 0) is displayed as 1 on screen, empty cell as 0
        # if player == 1:
        #     action = int(input("Enter your move (0-8): "))  # human player's input for cell number
        if player == 1:
            while True:
                try:
                    action = int(input("Enter your move (0-8): "))  # human player's input for cell number

                    if action < 0 or action > 8:
                        print("Invalid input. Enter a number from 0 to 8.")
                        continue

                    board_state = env.board
                    if board_state[action] != 0:
                        print("That cell is already taken. Choose an empty cell.")
                        continue

                    break
                except ValueError:
                    print("Invalid input. Enter an integer from 0 to 8.")
        else:
            time.sleep(1)  # trained network's decision
            action = act(
                model,
                torch.tensor(np.array([obs]), dtype=torch.float).to(device),
            ).item()

        obs, _, done, exp = env.step(action)
        player = 1 - player  # change player turn from 0 to 1 or 1 to 0

    os.system("cls" if os.name == "nt" else "clear")
    print(
        "Commands:\n{}|{}|{}\n-----\n{}|{}|{}\n-----\n{}|{}|{}\n\nBoard:".format(
            *[x for x in range(0, 9)]
        )
    )
    env.render()
    print(exp)

    if "reason" in exp and "tied" in exp["reason"]:
        print("A tied game. ---------.")
        exit(0)


def load_model(path: str, device: torch.device):  # load trained model
    model = Network(n_inputs=3 * 9, n_outputs=9).to(device)
    checkpoint_data = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint_data["state_dict"])
    model.eval()
    return model


# get action from trained network
# def act(model: Network, state: torch.tensor):
#     with torch.no_grad():
#         p = F.softmax(model.forward(state), dim=-1).cpu().numpy()  # 9 outputs from network

#         # find non-empty cells
#         # following produces 9 values with True or False, False for non-empty cell
#         valid_moves = (
#             state.cpu().numpy().reshape(3, 3, 3).argmax(axis=2).reshape(-1) == 0
#         )

#         p = valid_moves * p  # 9 numbers with 0 where the cell is not empty
#         return p.argmax()  # best decision (0-8) on non-empty cells
def act(model: Network, state: torch.tensor):
    with torch.no_grad():
        q = model.forward(state).clone()  # 1 x 9

        # find non-empty cells
        # following produces 9 values with True or False, True for empty cell
        valid_moves = (
            state.cpu().numpy().reshape(3, 3, 3).argmax(axis=2).reshape(-1) == 0
        )

        invalid_indices = np.where(~valid_moves)[0]
        q[0, invalid_indices] = -1e9  # block illegal moves

        return q.max(1)[1].view(1, 1)

def main():
    #res = train()  # -------uncomment this line to train network

    # after model is trained, use the following code to play
    # human is player 2, select the cell number 0-8 to decide which cell
    # you want for your turn
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("checkpoint/tictactoe_policy_model_without_illegal_moves.pt", device)
    play(model)  # play one game against trained network


if __name__ == "__main__":
    sys.exit(int(main() or 0))