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
from core import ReplayMemory, Transition, Trajectory, SavedAction
import datetime
import os


experiment_config = {'name': 'CONST_BASELINE_delay_2_rew_10',
                     'timestamp': datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S%f"),
                     'x_label': 'Epoch',
                     'y_label': 'Sum of reward on the trajectory'}


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """

    def __init__(self):
        super(Policy, self).__init__()
        self.linear = nn.Linear(INP_DIM, 128)

        # actor's layer
        self.action = nn.Linear(128, N_ACTIONS)

        # critic's layer
        self.value = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.linear(x))
        action_prob = F.softmax(self.action(x), dim=-1)
        state_values = self.value(x)

        return action_prob, state_values


class LearnerActorCritic:
    def __init__(self, gamma=0.999,
                 log_interval=10,
                 num_episodes=NUM_EPISODES,
                 fix_delay=2,
                 is_random_delay=False,
                 reward_level=20,
                 custom_func=None,
                 custom_func_args=None
                 ):
        self.env = ChainAgent(inventory_level=10,
                              fix_delay=fix_delay,
                              reward_level=reward_level,
                              max_num_steps=MAX_STEPS + 10,
                              demand_generation_function=self.demand_generation_function,
                              custom_func=custom_func,
                              custom_func_args=custom_func_args,
                              is_random_delay=is_random_delay)

        self.gamma = gamma
        self.log_interval = log_interval
        self.num_episodes = num_episodes
        self.model = Policy()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.eps = np.finfo(np.float32).eps.item()
        self.rewards = []


    def demand_generation_function(self):
        '''
        Function for generating demand. Should be overwritten for custom demand.
        :return: int:
        '''
        return np.random.randint(0, 10)



    def select_action(self, state):
        state = torch.from_numpy(np.array(state)).float()
        probs, state_value = self.model(state)
        m = Categorical(torch.Tensor([1 if n == 5 else 0 for n in range(N_ACTIONS)]))
        # m = Categorical(torch.Tensor([1./N_ACTIONS for n in range(N_ACTIONS)]))
        action = m.sample()
        self.model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()


    def finish_episode(self):
        """
        Training code. Calcultes actor and critic loss and performs backprop.
        """
        R = 0
        saved_actions = self.model.saved_actions
        policy_losses = []
        value_losses = []
        returns = []

        # calculate the true value using rewards returned from the environment
        for r in self.model.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.model.rewards[:]
        del self.model.saved_actions[:]


    rews = []
    def run(self):
        running_reward = 0 #TODO: was 10

        # run inifinitely many episodes
        for i_episode in range(self.num_episodes):

            # reset environment and episode reward
            state = self.env.reset()
            ep_reward = 0

            for step in range(100):

                # select action from policy
                action = self.select_action(state)

                # take the action
                state, reward, done, _ = self.env.step(action)
                # print(reward)
                # print('IL', self.env.inventory_level)
                # self.rews.append(reward)

                self.model.rewards.append(reward)
                ep_reward += reward

                if done:
                    break

            # update cumulative reward with smoothing
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            self.rewards.append(running_reward)

            # perform backprop
            self.finish_episode()

            # log results
            if i_episode % self.log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                      i_episode, ep_reward, running_reward/MAX_STEPS))



os.makedirs(experiment_config['name'], exist_ok=True)


# def f(x, alpha=20):
#     if np.abs

learner = LearnerActorCritic(num_episodes=10000,
                             reward_level=10,
                             fix_delay=3,
                             is_random_delay=True,
                             custom_func=lambda x, alpha: 0 if np.abs(x) > alpha else 1,
                             custom_func_args={'alpha': 10})
learner.run()

path = experiment_config['name']
torch.save(learner.model.state_dict(), os.path.join(path, experiment_config['timestamp'] + 'model'))

plt.plot(np.array(learner.rewards))

plt.title(experiment_config['name'])
plt.xlabel(experiment_config['x_label'])
plt.ylabel(experiment_config['y_label'])

np.save(os.path.join(path, experiment_config['timestamp'] + 'raw_data'), np.array(learner.rewards))
plt.savefig(os.path.join(path, experiment_config['timestamp'] + 'pic'))
plt.show()

torch.save(learner.model.state_dict(), os.path.join(path, experiment_config['timestamp'] + 'model'))