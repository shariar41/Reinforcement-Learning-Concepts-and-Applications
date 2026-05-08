import sys
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import torch
from DQN import DQN
import torch.optim as optim
from ReplayMemory import ReplayMemory
import torch.nn as nn
import os
from time import sleep
import numpy as np

env = gym.make("CartPole-v1")
plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
episode_durations = []

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
steps_done = 0


def main():
    #res = train()   # uncomment this line to train network

    # after model is trained, use the following code to play
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model('checkpoint/CartPoleDoubleDQN_model.pt', device)
    play(model)


def load_model(path: str, device: torch.device):
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

    model = DQN(n_observations, n_actions).to(device)
    checkpoint_data = torch.load(path)
    model.load_state_dict(checkpoint_data['state_dict'])
    model.eval()
    return model


def play(model):
    env = gym.make('CartPole-v1', render_mode="human")
    state, info = env.reset()

    for _ in range(200):
        action = select_action_play(model, state)
        state, reward, done, _,_ = env.step(action)
        env.render()
        sleep(0.1)

    env.close()


def train():
    global steps_done

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon
    # TAU is the update rate of the target network
    # LR is the learning rate of the AdamW optimizer

    TAU = 0.005
    LR = 1e-4

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    if torch.cuda.is_available():
        num_episodes = 1000
    else:
        num_episodes = 700

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action = select_action(state, policy_net)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of optimization
            optimize_model(memory, policy_net, target_net, optimizer)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 − τ) θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = (
                    policy_net_state_dict[key] * TAU
                    + target_net_state_dict[key] * (1 - TAU)
                )

            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()

    print("----------saving model in file-----------------")
    checkpoint_data = {
        'epoch': num_episodes,
        'state_dict': policy_net.state_dict(),
    }

    ckpt_path = os.path.join("checkpoint", "CartPoleDoubleDQN_model.pt")
    torch.save(checkpoint_data, ckpt_path)


def select_action_play(model, state):
    if len(state) != 4:
        state = state[0]

    state = torch.tensor(state, dtype=torch.float32, device=device)

    with torch.no_grad():
        values = model(state)

    action = np.argmax(values.cpu().numpy())
    return action


def select_action(state, policy_net):
    global steps_done

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)

    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')

    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)


def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    # Transpose the batch
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool
    )

    non_final_next_states_list = [s for s in batch.next_state if s is not None]

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    if len(non_final_next_states_list) > 0:
        non_final_next_states = torch.cat(non_final_next_states_list)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # Compute expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if __name__ == "__main__":
    sys.exit(int(main() or 0))