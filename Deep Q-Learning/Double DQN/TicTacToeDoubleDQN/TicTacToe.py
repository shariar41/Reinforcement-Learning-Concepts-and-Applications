import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TicTacToe(gym.Env):
    """
    Tictac Toe cell indices:
    [0, 1, 2,
     3, 4, 5,
     6, 7, 8]
    """

    def __init__(self, summary: dict = None):
        super().__init__()

        if summary is None:
            summary = {
                "total games": 0,
                "ties": 0,
                "illegal moves": 0,
                "player 0 wins": 0,
                "player 1 wins": 0,
            }

        self.summary = summary
    reward_range = (-np.inf, np.inf)
    observation_space = spaces.MultiDiscrete([2 for _ in range(0, 9 * 3)])
    # required by gym (not used in the code we write) - one hot encoding of each cell
    action_space = spaces.Discrete(9)  # needed by gym, set of actions 0-8

    winning_streaks = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ]

    def seed(self, seed=None):
        pass

    def _one_hot_board(self):  # encodes state as 27 numbers, each cell's state
        # human player (player 1) is displayed as state 2 in the cell
        # network (player 0) is displayed as 1 on screen, empty cell as 0
        if self.current_player == 0:  # is one hot encoded, 1 0 0 => empty
            # print(self.board) # 0 1 0 => player 1 (displayed as 2) i.e., human
            # 0 0 1 => player 0 (network)
            return np.eye(3)[self.board].reshape(-1)  # eye=>diagonals are 1

        if self.current_player == 1:
            # permute for symmetry
            # self.board = [0,0,1,0,2,0,1,2,0] - example
            # above board state produces: [1 0 0 1 0 0 0 0 1 1 0 0 0 1 0 1 0 0
            # 0 0 1 0 1 0 1 0 0]
            # print(np.eye(3)[self.board][:, [0, 2, 1]].reshape(-1))
            return np.eye(3)[self.board][:, [0, 2, 1]].reshape(-1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 0
        self.board = np.zeros(9, dtype="int")  # all cells are empty
        return self._one_hot_board(), {}

    def step(self, actions):
        exp = {"state": "in progress"}
        # get the current player's action
        action = actions
        reward = 0
        done = False

        # illegal move
        if self.board[action] != 0:  # not 0 implies cell was occupied, so illegal move
            reward = -10  # illegal moves are really bad
            exp = {"state": "done", "reason": "Illegal move"}
            done = True
            self.summary["total games"] += 1
            self.summary["illegal moves"] += 1

            # _one_hot_board encodes the current state as 27 values
            # one hot for each of the 9 cells
            return self._one_hot_board(), reward, done, exp

        # if current_player is 0, it is player 1's turn and vice versa
        self.board[action] = self.current_player + 1

        # check if the player 1 can win on the next turn:
        for streak in self.winning_streaks:
            # zz = self.board[streak]
            # check for 1 or 2 entry in board, as 0 means empty cell
            if ((self.board[streak] == 2 - self.current_player).sum() >= 2) and (
                self.board[streak] == 0
            ).any():
                # example board=[0 2 0 0 1 0 1 2 0], streak = [2 4 6] i.e., diagonal
                reward = -2  # player 1 is human player (indicated by 2 in cell),
                exp = {  # pos. 2 is empty that human player can take to win
                    "state": "in progress",
                    "reason": "Player {} can lose on the next turn".format(
                        self.current_player
                    ),
                }

        # check if network won (player 0 indicated by 1 in cell)
        for streak in self.winning_streaks:
            if (self.board[streak] == self.current_player + 1).all():  # satisfies streak
                reward = 1  # network wins streak=[0,4,8] diagonal 1's
                # example board = [1, 2, 0, 0, 1, 2, 1, 2, 1], network is player 0 (indicated by 1)
                exp = {
                    "state": "in progress",
                    "reason": "Player {} has won".format(self.current_player),
                }
                self.summary["total games"] += 1
                self.summary["player {} wins".format(self.current_player)] += 1
                done = True

        # check if tied, which ends the game
        if (self.board != 0).all() and not done:
            reward = 0
            exp = {
                "state": "in progress",
                "reason": "Player {} has tied".format(self.current_player),
            }
            done = True
            self.summary["total games"] += 1
            self.summary["ties"] += 1

        # move to the next player
        self.current_player = 1 - self.current_player
        
        return self._one_hot_board(), reward, done, exp

    def render(self, mode: str = "human"):
        print("{}|{}|{}\n-----\n{}|{}|{}\n-----\n{}|{}|{}".format(*self.board.tolist()))