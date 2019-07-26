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
import gym
from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000

from env import ChainAgent


fr_lake = gym.make('FrozenLakeNotSlippery-v0')

max_steps = 100
INP_DIM = fr_lake.observation_space.n
n_actions = fr_lake.action_space.n

def demand_generation_function():
    return np.random.randint(0, 10)

env = ChainAgent(inventory_level=10,
                   fix_delay=1,
                   max_num_steps=max_steps+10,
                   demand_generation_function=demand_generation_function)

env = fr_lake

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(INP_DIM, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, n_actions)


    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        # x = self.lin3(x)
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
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
TARGET_UPDATE = 1

# init_screen = get_screen()
# _, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
# n_actions = env.action_space.n

policy_net = DQN().to(device)
# target_net = DQN().to(device)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    state = torch.Tensor(state)[None, :]
    # print(state)
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        print('greedy')
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        print('random')
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

# def select_action(state):
#     return torch.LongTensor([[np.random.randint(0, 4)]])

episode_durations = []
s = env.reset()

def to_one_hot(position, n=fr_lake.observation_space.n):
    output = np.zeros(n)
    output[int(position)] = 1
    return output

s = to_one_hot(s)

# print(len(s))

def optimize_model():

    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # for i in batch.next_state:
        # print(i.shape)

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # print('st batch shape', state_batch.shape)
    # print('act batch shape', action_batch.shape)
    # print('NST;', batch.next_state)
    next_states = torch.cat(batch.next_state)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # print(state_action_values)

    # print('SAV:', state_action_values.shape)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # next_state_values = target_net(next_states).max(1)[0].detach()
    next_state_values = policy_net(next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # print('rewards:', reward_batch)
    # print('ESAV:', expected_state_action_values)

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # print(loss)
    # loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()


rewards = []

num_episodes = 10000
trajectory_len = 100
for i_episode in range(num_episodes):
    state = to_one_hot(env.reset())
    for step in range(trajectory_len):
        # print('st:', state)
        action = select_action(torch.Tensor(state))
        next_state, reward, done, _ = fr_lake.step(action.item())

        rewards.append(reward)
        next_state = to_one_hot(next_state)
        print('PROBS', policy_net(torch.Tensor(state)))
        print('State', state)
        # print('action:', action)
        # print('nxst:', next_state)
        # print('del:', env.)

        # print('reward:', reward)
        # env.render(0)
        # print('NS:', len(next_state))
        reward = torch.tensor([reward], device=device)

        memory.push(torch.Tensor([state]), action, torch.Tensor([next_state]), reward)
        state = next_state

        optimize_model()
        if done:
            print()
            print()
            if reward > 0:
                env.render()
                ...
            break


hole_state = torch.Tensor([0 for _ in range(16)])
for i in range(16):
    hole_state[i] = 1
    print(policy_net(hole_state))
    hole_state[i] = 0






    # print()



# plt.plot(rewards)
# plt.show()
# plt.scatter(range(len(env.r)), np.array(env.r))
# plt.hist(env.r, bins=50)
# plt.hist(env.d, bins=50)
# plt.hist(env.dif, bins=50)
# plt.plot(range(len(env.dif)), np.array(env.dif))

# print(sum(env.r)/len(env.r))
# print(env.r)

# plt.plot(env.il)
# plt.plot(env.dif)
# plt.show()
# print(max(env.d))
# print(min(env.r))
# print(min(env.d))
# print(sum(env.d)/len(env.d))
# print(sum(env.dif)/len(env.dif))
# print((np.array(env.dif) >= 0).sum())
    # if i_episode % TARGET_UPDATE == 0:
        # print(reward)
        # target_net.load_state_dict(policy_net.state_dict())

# from catalyst.rl.offpolicy.algorithms import DDPG
# ddpg = DDPG()