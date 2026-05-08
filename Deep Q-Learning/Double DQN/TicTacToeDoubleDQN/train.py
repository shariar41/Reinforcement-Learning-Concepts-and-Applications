from TicTacToe import TicTacToe
from Network import Network
from ReplayMemory import Transition, ReplayMemory
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import random
import logging
import io


def train(
    n_steps: int = 100_000,  # 500_000
    batch_size: int = 128,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.1,
    eps_steps: int = 200_000,
) -> bytes:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Beginning training on: {}".format(device))
    logging.info("Beginning training on: {}".format(device))

    target_update = int((1e-2) * n_steps)  # target network copy interval

    policy = Network(n_inputs=3 * 9, n_outputs=9).to(device)  # Q network
    target = Network(n_inputs=3 * 9, n_outputs=9).to(device)  # separate target network
    target.load_state_dict(policy.state_dict())  # copy policy net to target net
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    memory = ReplayMemory(50_000)  # Experience store
    
    env = TicTacToe()
    state, _ = env.reset()
    state = torch.tensor(np.array([state]), dtype=torch.float).to(device)

    old_summary = {
        "total games": 0,
        "ties": 0,
        "illegal moves": 0,
        "player 0 wins": 0,
        "player 1 wins": 0,
    }

    _randoms = 0
    summaries = []

    for step in range(n_steps):
        t = np.clip(step / eps_steps, 0, 1)
        eps = (1 - t) * eps_start + t * eps_end  # eps is gradually reduced
        # more random actions in the beginning, gradually making less random
        # by the network
        #
        # get action from network - player 1
        action, was_random = select_model_action(device, policy, state, eps)

        if was_random:
            _randoms += 1

        next_state, reward, done, _ = env.step(action.item())

        # player 2 - during training, player 2 makes random moves
        if not done:
            next_state, _, done, _ = env.step(select_dummy_action(next_state))

        next_state = torch.tensor(np.array([next_state]), dtype=torch.float).to(device)

        if done:
            next_state = None

        # next_state is decided by our action on current state, followed
        # by other player (in this case random)'s action
        memory.push(state, action, next_state, torch.tensor([reward], device=device))

        state = next_state

        optimize_model(
            device=device,
            optimizer=optimizer,
            policy=policy,
            target=target,
            memory=memory,
            batch_size=batch_size,
            gamma=gamma,
        )

        if done:
            state, _ = env.reset()
            state = torch.tensor(np.array([state]), dtype=torch.float).to(device)

        if step % target_update == 0:
            target.load_state_dict(policy.state_dict())  # copy policy net to target

        if step % 5000 == 0:
            delta_summary = {k: env.summary[k] - old_summary[k] for k in env.summary}
            delta_summary["random actions"] = _randoms
            old_summary = {k: env.summary[k] for k in env.summary}
            logging.info("{} : {}".format(step, delta_summary))
            print("{} : {}".format(step, delta_summary))
            summaries.append(delta_summary)
            _randoms = 0

    logging.info("Complete")

    res = io.BytesIO()
    torch.save(policy.state_dict(), res)

    print("----------saving model in file-----------------")
    checkpoint_data = {
        "epoch": n_steps,
        "state_dict": policy.state_dict(),
    }

    ckpt_path = os.path.join("checkpoint/tictactoe_policy_model_without_illegal_moves.pt")
    torch.save(checkpoint_data, ckpt_path)

    return res.getbuffer()


def optimize_model(  # obtain a batch of data from memory i.e.,experience store
    device: torch.device,
    optimizer: optim.Optimizer,
    policy: Network,  # for computing Q(s,a)
    target: Network,  # for computing target Q(s,a) values
    memory: ReplayMemory,
    batch_size: int,
    gamma: float,
):
    """from the Torch DQN tutorial.
    Arguments:
    device {torch.device} -- Device
    optimizer {torch.optim.Optimizer} -- Optimizer
    policy {Policy} -- Policy module
    target {Policy} -- Target module
    memory {ReplayMemory} -- Replay memory
    batch_size {int} -- Number of observations to use per batch step
    gamma {float} -- discount factor for reward
    """
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    # each of the four fields in batch i.e., Transition is an array of batch-size

    # Compute a mask of non-final states (where next state is not None)
    # and concatenate the batch elements
    non_final_mask = torch.tensor(  # tensor of bools, False if next_state is None
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )

    non_final_next_states  =  torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)  # batch_sizex27
    action_batch = torch.cat(batch.action)  # batch_sizex1
    reward_batch = torch.cat(batch.reward)  # batch_sizex1

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy(state_batch).gather(1, action_batch)

    # In above code policy(state_batch) produces batch_sizex9
    # then gather(1,action_batch) selects the one of the 9 outputs
    # according to action taken, thus producing output of batch_sizex1
    # i.e., Q(s,a) values for one action

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)

    next_state_values[non_final_mask] = target(non_final_next_states).max(1)[0].detach()

    # above code computes the max of (Q(next_state, all actions))
    # and stores these values for those cases where the next_state was not None
    # otherwise a value of 0 is left in the batch_size array

    # Compute the target Q values
    target_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(
        state_action_values, target_state_action_values.unsqueeze(1)
    )

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def select_dummy_action(state: np.array) -> int:
    """Select a random (valid) move, given a board state.
    Arguments:
    state {np.array} -- Board state observation
    Returns:
    int -- Move to make.
    """
    state = state.reshape(3, 3, 3)
    open_spots = state[:, :, 0].reshape(-1)
    valid_indices = np.where(open_spots == 1)[0]
    # p = open_spots / open_spots.sum()
    return np.random.choice(valid_indices)#np.arange(9), p=p


def select_model_action(
    device: torch.device, model: Network, state: torch.tensor, eps: float
) -> Tuple[torch.tensor, bool]:
    """Selects an action for the model: either using the policy, or
    by choosing a random valid action (as controlled by `eps`)
    Arguments:
    device {torch.device} -- Device
    model {Policy} -- Policy module
    state {torch.tensor} -- Current board state, as a torch tensor
    eps {float} -- Probability of choosing a random state.
    Returns:
    Tuple[torch.tensor, bool] -- The action, and a bool indicating whether
    the action is random or not.
    """
    # find valid moves from one-hot encoded board
    valid_moves = (
        state.cpu().numpy().reshape(3, 3, 3).argmax(axis=2).reshape(-1) == 0
    )
    valid_indices = np.where(valid_moves)[0]

    sample = random.random()
    sample = random.random()

    if sample > eps:
        with torch.no_grad():
            q_values = model(state).clone()  # 1 x 9
            invalid_indices = np.where(~valid_moves)[0]
            q_values[0, invalid_indices] = -1e9  # mask illegal moves
            action = q_values.max(1)[1].view(1, 1)
        return action, False
        # return model.act(state), False
    else:
        return (
            torch.tensor(
                #[[random.randrange(0, 9)]],
                [[random.choice(valid_indices.tolist())]],
                device=device,
                dtype=torch.long,
            ),
            True,
        )
# fix it so it only selects valid moves
# def select_model_action(
#     device: torch.device, model: Network, state: torch.tensor, eps: float
# ):
#     sample = random.random()

#     # Get valid moves (empty cells)
#     state_np = state.cpu().numpy().reshape(3,3,3)
#     valid_moves = (state_np.argmax(axis=2).reshape(-1) == 0)
#     valid_indices = np.where(valid_moves)[0]

#     if sample > eps:
#         with torch.no_grad():
#             q_values = model(state).cpu().numpy().reshape(-1)

#             # Mask invalid moves
#             q_values[~valid_moves] = -1e9

#             action = np.argmax(q_values)
#             return torch.tensor([[action]], device=device, dtype=torch.long), False
#     else:
#         action = random.choice(valid_indices)
#         return torch.tensor([[action]], device=device, dtype=torch.long), True