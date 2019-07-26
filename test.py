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
from constants import *
from env import ChainAgent

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(INP_DIM, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


class LearnerDQN:
    def __init__(self, clip_grad=True, num_episodes=50, trajectory_len=MAX_STEPS):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_nn()
        self.num_episodes = num_episodes
        self.trajectory_len = trajectory_len
        self.model = DQN()
        self.replay = ReplayMemory(10000)
        self.steps_done = 0
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.clip_grad=clip_grad
        self.env = ChainAgent(inventory_level=10,
                         fix_delay=1,
                         max_num_steps=MAX_STEPS + 10,
                         demand_generation_function=self.demand_generation_function)

    def select_action(self, state):
        state = torch.Tensor(state)[None, :]
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            # print('greedy')
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # print('random')
            return torch.tensor([[random.randrange(N_ACTIONS)]], device=self.device, dtype=torch.long)


    def demand_generation_function(self):
        return np.random.randint(0, 10)


    def optimize_model(self):

        if len(self.replay) < BATCH_SIZE:
            return

        transitions = self.replay.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values = self.target_net(next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        if self.clip_grad:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def build_nn(self):
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


    def run(self):
        for i_episode in range(self.num_episodes):
            state = self.env.reset()
            for step in range(self.trajectory_len):
                action = self.select_action(torch.Tensor(state))
                next_state, reward, done, _ = self.env.step(action.item())
                print('R:', reward)
                print('IL:', self.env.inventory_level)
                reward = torch.tensor([reward], device=self.device)
                self.replay.push(torch.Tensor([state]), action, torch.Tensor([next_state]), reward)
                state = next_state
                self.optimize_model()

                if done:
                    break

        if i_episode % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())



learner = LearnerDQN(num_episodes=1000, trajectory_len=100)
learner.run()
