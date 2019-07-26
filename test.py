import numpy as np
import torch.nn as nn
from collections import namedtuple
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


INP_DIM = 16
N_ACTIONS = 4

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(INP_DIM, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, N_ACTIONS)


    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        # x = F.relu(self.lin3(x))
        x = self.lin3(x)
        return x


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import gym
from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)


fr_lake = gym.make('FrozenLakeNotSlippery-v0')


fr_lake.reset()
fr_lake.step(1)
fr_lake.step(2)
print(fr_lake.step(2))
fr_lake.render()

