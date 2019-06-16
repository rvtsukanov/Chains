import numpy as np
import torch.nn as nn
from collections import namedtuple
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

from env import ChainAgent

max_steps = 90
INP_DIM = 8
n_actions = 10

def demand_generation_function():
    return np.random.randint(10, 20)

env = ChainAgent(inventory_level=10,
                   fix_delay=1,
                   max_num_steps=max_steps+10,
                   demand_generation_function=demand_generation_function)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(INP_DIM, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, n_actions)


    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
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

BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# init_screen = get_screen()
# _, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
# n_actions = env.action_space.n

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    state = torch.Tensor(state)[None, :]
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = []
env.reset()


def optimize_model():

    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # print(batch.state)
    # print(torch.cat(batch.state), 1)

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_states = torch.cat(batch.next_state)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # print(state_action_values)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values = target_net(next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 50
trajectory_len = max_steps

for i_episode in range(num_episodes):
    state = env.reset()
    for step in range(trajectory_len):
        action = select_action(torch.Tensor(state))
        next_state, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        memory.push(torch.Tensor([state]), action, torch.Tensor([next_state]), reward)
        print(action)
        state = next_state

        optimize_model()
        if done:
            break

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

