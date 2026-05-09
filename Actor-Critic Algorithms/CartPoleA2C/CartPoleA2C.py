import sys
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
from Network import Network
from ActorCritic import ActorCritic
import torch.optim as optim
import torch.nn as nn
import os
from time import sleep
import numpy as np
import torch.nn.functional as F
import torch.distributions as distributions

env = gym.make("CartPole-v1")
test_env = gym.make('CartPole-v1')

#plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPISODES = 500
GAMMA = 0.99 # discount factor
N_TRIALS = 25
REWARD_THRESHOLD = 400
PRINT_EVERY = 10
LR = 0.001

train_rewards = []
test_rewards = []

def main():
    #n_actions = env.action_space.n
    ## Get the number of state observations
    #state, info = env.reset()
    #n_observations = len(state)
    
    #actor = Network(n_observations, n_actions).to(device)
    #critic = Network(n_observations, 1).to(device) # critic has single output for V
    
    #policy_net = ActorCritic(actor, critic).to(device)
    
    #optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    
    #res = train(policy_net, actor, critic, optimizer) # -------uncomment this line to train network
    
    #print("----------saving model in file-----------------")
    #checkpoint_data_policy = {
    #    'epoch': MAX_EPISODES,
    #    'state_dict': policy_net.state_dict(),
    #}
    #ckpt_path = os.path.join("checkpoint/CartPoleA2C_policy_model.pt")
    #torch.save(checkpoint_data_policy, ckpt_path)
    
    # after model is trained, use the following code to play
    model = load_model('checkpoint/CartPoleA2C_policy_model.pt',device)
    play(model) # play one game against trained network

def load_model(path: str, device: torch.device): # load trained model
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)
    actor = Network(n_observations, n_actions).to(device)
    critic = Network(n_observations, 1).to(device) # critic has single output for V
    model = ActorCritic(actor, critic).to(device)
    checkpoint_data = torch.load(path)
    model.load_state_dict(checkpoint_data['state_dict'])
    model.eval()
    return model

def play(model):
    #----------test trained model-----------
    env = gym.make('CartPole-v1',render_mode="human")
    env.reset()
    state = env.reset()
    for _ in range(200):
        action = select_action_play(model, state)
        state, reward, done, _,_ = env.step(action)
        env.render()
        sleep(0.1)
    env.close()
    
def train(policy_net, actor, critic, optimizer):
    for episode in range(1, MAX_EPISODES+1):
        policy_loss, value_loss, train_reward = train_one_episode(policy_net,
        actor, critic, optimizer, GAMMA)
        test_reward = compute_reward_one_episode(policy_net)
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)
        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
        if episode % PRINT_EVERY == 0:
            print(f'| Episode: {episode:3} | Mean Train Rewards:{mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
        if mean_test_rewards >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            break
    plot_rewards(train_rewards, test_rewards)

def compute_reward_one_episode(policy_net):
    policy_net.eval()
    done = False
    episode_reward = 0
    state = test_env.reset()[0]
    
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_pred,_ = policy_net(state)
            action_prob = F.softmax(action_pred, dim = -1)
        action = torch.argmax(action_prob, dim = -1)

        state, reward, done, _, _ = test_env.step(action.item())
        episode_reward += reward
    return episode_reward
    
def train_one_episode(policy_net, actor, critic, optimizer, discount_factor):
    policy_net.train()
    log_prob_actions = []
    rewards = []
    values = []
    done = False
    episode_reward = 0
    state = env.reset()[0]
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_pred = actor(state)
        value_pred = critic(state)
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample() # will return 1 or 0 - for CartPole, only two actions
        log_prob_action = dist.log_prob(action)
        state, reward, done, _, _ = env.step(action.item())
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        episode_reward += reward
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)
    discounted_rewards = calculate_discounted_rewards(rewards, discount_factor)
    advantages = calculate_advantages(discounted_rewards, values)
    policy_loss, value_loss = update_policy(advantages, log_prob_actions,discounted_rewards, values, optimizer)
    return policy_loss, value_loss, episode_reward
    
def calculate_advantages(discounted_rewards, values, normalize = True):
    advantages = discounted_rewards.to(device) - values.to(device)
    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages
    
def calculate_discounted_rewards(rewards, discount_factor, normalize = True):
    returns = [] # apply the discounted rewards to the episode state actions
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns)
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
    return returns

def update_policy(advantages, log_prob_actions, discounted_rewards, values,
    optimizer):
    # updates the policy network
    policy_loss = - (advantages * log_prob_actions).sum()
    # - sign makes the maximization problem to minimization
    value_loss = F.smooth_l1_loss(discounted_rewards.to(device), values.to(device)).sum()
    
    optimizer.zero_grad()
    policy_loss.backward(retain_graph=True)
    value_loss.backward()
    
    optimizer.step()
    return policy_loss.item(), value_loss.item()
    
def plot_rewards(train_rewards,test_rewards):
    plt.figure(figsize=(12,8))
    plt.plot(test_rewards, label='Test Reward')
    plt.plot(train_rewards, label='Train Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Reward', fontsize=20)
    plt.hlines(REWARD_THRESHOLD, 0, len(test_rewards), color='r')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    
def select_action_play(model, state):
    if len(state) != 4:
        state = state[0]
    state = torch.Tensor(state).to(device) # added [0]
    with torch.no_grad():
        action_pred,_ = model(state)
        action_prob = F.softmax(action_pred, dim = -1)
    action = np.argmax(action_prob.cpu().numpy())
    return action

if __name__ == "__main__":
    sys.exit(int(main() or 0))