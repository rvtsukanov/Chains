import numpy as np
import torch.nn as nn
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import math
import matplotlib.pyplot as plt
from constants import *
from env import ChainAgent
from core import ReplayMemory, Transition, Trajectory
import datetime
import os


experiment_config = {'name': 'REINFORCE_delay_2_rew10',
                     'timestamp': datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S%f"),
                     'x_label': 'Epoch',
                     'y_label': 'Sum of reward on the trajectory'}


class Actor(nn.Module):
    '''
    Actor's base class
    '''
    def __init__(self):
        super(Actor, self).__init__()
        self.lin1 = nn.Linear(INP_DIM, INNER_DIM)
        self.lin2 = nn.Linear(INNER_DIM, INNER_DIM)
        self.lin3 = nn.Linear(INNER_DIM, N_ACTIONS)
        self.softmax = nn.Softmax() # Probably, just logsoftmax


    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.softmax(self.lin3(x))
        return x


class LearnerREINFORCE:
    '''
    Learner class - abstraction which includes configuration of experiment, necessary models, and all actions needed for conduction.
    '''
    def __init__(self, num_episodes=NUM_EPISODES, trajectory_len=MAX_STEPS, clip_grad=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_episodes = num_episodes
        self.trajectory_len = trajectory_len
        self.replay = ReplayMemory(10000)
        self.steps_done = 0
        self.lr = 1e-3
        self.actor = Actor()
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.clip_grad = clip_grad
        self.trajectory = []
        self.trajectories = []
        self.optimize_each = 5
        self.gamma = 0.99

        self.env = ChainAgent(inventory_level=10,
                              fix_delay=1,
                              max_num_steps=MAX_STEPS + 10,
                              demand_generation_function=self.demand_generation_function)


    def demand_generation_function(self):
        return np.random.randint(0, 10)


    def choose_action(self, state):
        state = torch.FloatTensor(state)
        policy_output = self.actor(state)
        c = Categorical(policy_output)
        return c.sample(), policy_output

    @staticmethod
    def discount_and_norm_rewards(episode_rewards, gamma=1):
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        cumulative = 0
        for t in reversed(range(len(episode_rewards))):
            cumulative = cumulative * gamma + episode_rewards[t]
            discounted_episode_rewards[t] = cumulative
        return discounted_episode_rewards


    def optimize_model(self):

        states = []
        rewards = []
        policies = []

        for s, a, r, p in self.trajectory:
            states.append(s)
            rewards.append(r)
            policies.append(p)

        discounted_rewards = torch.Tensor(self.discount_and_norm_rewards(np.array(rewards)))
        print(discounted_rewards)


        for n, (s, a, r, p) in enumerate(self.trajectory):
            self.optimizer.zero_grad()
            # print(policies)
            pseudo_loss = -self.lr * torch.log(p) * discounted_rewards[n]
            pseudo_loss.backward()
            for param in self.actor.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()



    def roll_trajectory(self):
        state = self.env.reset()
        self.trajectory = []
        for step in range(MAX_STEPS):
            action, policy_output = self.choose_action(state)
            print(policy_output)
            state, reward, done, meta = self.env.step(action)
            print(reward)
            self.trajectory.append((state, action, reward, policy_output[action]))

            if done:
                break

    def run(self):
        for episode in range(NUM_EPISODES):
            self.roll_trajectory()
            # if episode % self.optimize_each == 0:
            self.optimize_model()



learner = LearnerREINFORCE()

learner.run()

