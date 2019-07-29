import numpy as np
import torch.nn as nn
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from constants import *
from env import ChainAgent
from core import ReplayMemory, Transition
import datetime
import os

#Confing used in creating corresponding folders in root directory and name data/plots/models recieved from particular experiment
experiment_config = {'name': 'DQN_delay_2_rew_qudratic',
                     'timestamp': datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S%f"),
                     'x_label': 'Epoch',
                     'y_label': 'Sum of reward on the trajectory'}


class DQN(nn.Module):
    '''
    Deep Q-Network architecture.
    '''
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
    '''
    Learner class - abstraction which includes configuration of experiment, necessary models, and all actions needed for conduction.
    '''
    def __init__(self, clip_grad=True,
                 num_episodes=50,
                 trajectory_len=MAX_STEPS,
                 custom_func=None,
                 custom_func_args=None
                 ):
        '''
        Initialization
        :param clip_grad: bool: flag for clipping gradients with value 1
        :param num_episodes: number of episodes to run
        :param trajectory_len: maximal number of steps in each trajectory
        :param custom_func: custom reward function
        :param custom_func_args: custom reward function arguments
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_nn()
        self.num_episodes = num_episodes
        self.trajectory_len = trajectory_len
        self.model = DQN()
        self.replay = ReplayMemory(10000)
        self.steps_done = 0
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.clip_grad=clip_grad

        self.rewards = []
        self.modules = []
        self.env = ChainAgent(inventory_level=10,
                         fix_delay=1,
                         max_num_steps=MAX_STEPS + 10,
                         demand_generation_function=self.demand_generation_function,
                         custom_func=custom_func,
                         custom_func_args=custom_func_args)

    def select_action(self, state):
        '''
        Implementation of e-greedy approach
        :param state: input state to choose appropriate action
        :return: Tensor: action
        '''
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
        '''
        Default function to generate demand
        :return: int: demand level
        '''
        return np.random.randint(0, 10)


    def optimize_model(self):
        '''
        Method of optimizing parameters of neural net
        :return:
        '''

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
        # next_state_values = self.target_net(next_states).max(1)[0].detach()
        next_state_values = self.policy_net(next_states).max(1)[0].detach()

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
        '''
        Building torch graph
        :return:
        '''
        self.policy_net = DQN().to(self.device)
        # self.target_net = DQN().to(self.device)
        # self.target_net.load_state_dict(self.policy_net.state_dict())
        # self.target_net.eval()


    def get_stat(self, state, next, reward, action):
        '''
        Some visualisation
        :param state:
        :param next:
        :param reward:
        :param action:
        :return:
        '''
        print('=====')
        print('DEM: ', self.env.demand_next, 'ST: ', state, ' -> ', action)
        print('NXST: ', next, 'REW: ', reward)
        print('=====')
        print()


    def run(self):
        '''
        Main loop of training. Iterates over num_episodes * trajectory len steps
        :return:
        '''
        for i_episode in range(self.num_episodes):
            state = self.env.reset()
            rewards = 0
            for step in range(self.trajectory_len):
                action = self.select_action(torch.Tensor(state))
                next_state, reward, done, _ = self.env.step(action.item())
                reward *= 1.
                rewards += reward
                reward = torch.tensor([reward], device=self.device)
                self.replay.push(torch.Tensor([state]), action, torch.Tensor([next_state]), reward)
                state = next_state
                self.optimize_model()

                if done:
                    break

            self.rewards.append(rewards)

            if i_episode % TARGET_UPDATE == 0:
                print(i_episode, ' : ', np.array(self.rewards[-100:]).mean())


# Saving module

os.makedirs(experiment_config['name'], exist_ok=True)

learner = LearnerDQN(num_episodes=10000,
                     trajectory_len=100,
                     custom_func=lambda x, alpha: -(x ** 2)/alpha + 1,
                     custom_func_args={'alpha': 400})
learner.run()


plt.plot(np.array(learner.rewards))

plt.title(experiment_config['name'])
plt.xlabel(experiment_config['x_label'])
plt.ylabel(experiment_config['y_label'])

path = experiment_config['name']
np.save(os.path.join(path, experiment_config['timestamp'] + 'raw_data'), np.array(learner.rewards))
plt.savefig(os.path.join(path, experiment_config['timestamp'] + 'pic'))
plt.show()


torch.save(learner.policy_net.state_dict(), os.path.join(path, experiment_config['timestamp'] + 'model'))