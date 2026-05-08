import random
from collections import namedtuple

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
# namedtuple is like a light weight class with fields
# In this case 4 fileds. Can be invoked like a constructor


class ReplayMemory(object):
    """from PyTorch DQN tutorial.
    During training, observations from the replay memory are
    sampled for Q learning.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # In the following, args composes the parameters as tuple, * unpacks it.
    # we do not have to declare 4 parameters explicitly this way
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # randomly selected samples

    def __len__(self):
        return len(self.memory)