import torch
import numpy as np
from QModel import QModel

class Agent:
    """ The Agent employs a Q Model, uses the optimal Q-value after training, and
    epsilon greedy during training """
    def __init__(
        self, epsilon=0.2, learning_rate=0.01, gamma=0.9, player_id=1):
        """
        epsilon: float=0.2: probab of random move by Agent.
        Rest of the time, it will choose the move
        according to the Q value
        """
        self.player_id = player_id
        self.qmodel = QModel()
        self.learning_rate = learning_rate
        self._optimizer = torch.optim.Adam(self.qmodel.parameters(), lr=learning_rate)
        self.gamma = gamma # discount_factor
        self.epsilon = epsilon # for greedy epsilon during training
    def get_random_action(self): # 0 to 8 possible values for action
        return np.random.randint(0, 9)

    def get_Q_action(self, state): # pick action according to best Q value
        with torch.no_grad():
            state2d, turn = state # state is a tuple, turn is 1 or 2
            turns = torch.tensor(turn, dtype=torch.int64)[None] # adds dim for batch
            #mask = ~(state2d != 0)
            mask = ~(state2d != 0)*1 # empty cells will be true
            mask2 = (state2d != 0)*-1000
            states2d = torch.tensor(state2d, dtype=torch.int64)[None]
            qvalues = self.qmodel(states2d, turns)[0]
            masked_qactions = qvalues * mask.flatten() + mask2.flatten()
            action = np.argmax(masked_qactions)
            ax, ay = torch.div(action, 3, rounding_mode='trunc'), action % 3
            while state[0][ax, ay] != 0: # invalid move
                action = self.get_random_action()
                ax, ay = torch.div(action, 3, rounding_mode='trunc'), action % 3
        return np.argmax(masked_qactions)
    def get_epsilon_greedy_action(self, state):
        if np.random.rand() < self.epsilon:
            action = self.get_random_action()
            ax, ay = torch.div(action, 3, rounding_mode='trunc'), action % 3
            # check if board is already occupied at location
            while state[0][ax, ay] != 0: # cell is occupied
                action = self.get_random_action()
                ax, ay = torch.div(action, 3, rounding_mode='trunc'), action % 3
        else:
            action = self.get_Q_action(state)
            ax, ay = torch.div(action, 3, rounding_mode='trunc'), action % 3

            while state[0][ax, ay] != 0: # invalid move
                action = self.get_random_action()
                ax, ay = torch.div(action, 3, rounding_mode='trunc'), action % 3
        return action

    def do_Qlearning_on_agent_model(self, state_action_nstate_rewards):
        states, actions, next_states, rewards = zip(*state_action_nstate_rewards)
        states2d, turns = zip(*states)
        next_states2d, next_turns = zip(*next_states)
        turns = torch.tensor(turns, dtype=torch.int64)
        next_turns = torch.tensor(next_turns, dtype=torch.int64)
        states2d = torch.tensor(states2d, dtype=torch.int64)
        next_states2d = torch.tensor(next_states2d, dtype=torch.int64)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        with torch.no_grad():
            # get qvalues for current state of the game
            mask = (next_turns > 0).float() # wether the game is over or not
            next_qvalues = self.qmodel(next_states2d, next_turns)
            expected_qvalues_for_actions = rewards + (
            self.gamma * torch.max(next_qvalues, 1)[0])
            # update qvalues:
        qvalues_for_actions = torch.gather(
        self.qmodel(states2d, turns), dim=1, index=actions[:, None]).view(-1)
        loss = torch.nn.functional.smooth_l1_loss(
            qvalues_for_actions, expected_qvalues_for_actions
        )
        #loss = torch.nn.functional.mse_loss(
        # qvalues_for_actions, expected_qvalues_for_actions
        #)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.item()