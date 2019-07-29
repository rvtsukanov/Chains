# import torch
import numpy as np
import gym
import pandas as pd
from tabulate import tabulate
from prettytable import PrettyTable

class ChainAgent(gym.Env):
    '''
    Base class for simulation
    '''
    def __init__(self, inventory_level=0,
                 max_num_steps=15,
                 reward_level=20,
                 max_capability_of_storage=20,
                 delay_factor=0,
                 delay_mean=2,
                 delay_var=1,
                 min_demand=2,
                 max_demand=30,
                 fix_delay=3,
                 is_random_delay=False,
                 next_n_steps=10,
                 demand_generation_function=None,
                 custom_func=None,
                 custom_func_args=None
                 ):

        '''
        Initialization
        :param inventory_level: float: starting level of goods in stock
        :param max_num_steps: int:  length of a trajectory
        :param reward_level: int: level of inventory_level for recieving positive reward (=1)
        :param max_capability_of_storage: int: maximal number of goods in storage
        :param delay_factor: float:  probability of next order is delayed
        :param delay_mean: float: mean of delayed noise
        :param delay_var: float: variance of delayed noise
        :param min_demand: minimal value of demand
        :param max_demand: maximal value of demand
        :param fix_delay: int: fixed number of timesteps
        :param is_random_delay: bool: flag of using delay
        :param next_n_steps: number of steps in horizon to use as state
        :param demand_generation_function: function to generate demand
        :param custom_func: custom reward function
        :param custom_func_args: custom reward function args
        '''

        print('=========================================')
        print('INITIAL CONFIGURATION: inventory level {}'.format(inventory_level))
        print('=========================================')

        self.inventory_level = inventory_level
        self.max_num_steps = max_num_steps
        self.done = False
        self.c_IL_positive = 1
        self.is_random_delay = is_random_delay
        self.c_IL_negative = 1
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.fix_delay = fix_delay
        self.next_n_steps = next_n_steps
        self.custom_func = custom_func
        self.custom_func_args = custom_func_args
        self.demand_generation_function = demand_generation_function
        self.demand_next = self.demand_generation_function()
        self.reward_level = reward_level
        self.action_min = 0
        self.action_min = 10
        self.action_range = list(range(0, 10))
        self.time = -1
        self.ordered_goods = []
        self.recieved_goods = []
        self.upcoming_goods = np.zeros(self.max_num_steps)
        self.max_capability_of_storage = max_capability_of_storage
        self.delay_factor = delay_factor
        self.delay_mean = delay_mean
        self.delay_var = delay_var
        self.d = []
        self.r = []
        self.dif = []
        self.il = []
        self.max_shipment = 5


    def delayed(self):
        '''
        Function for generating delay in shipment:
        If flag is_random delay is on - use augmentation by addind normally distributed noise
        :return:
        '''

        if self.is_random_delay:
            if np.random.binomial(1, self.delay_factor):
                return self.fix_delay + np.random.normal(self.delay_mean, self.delay_var)
            else:
                return self.fix_delay
        else:
            return self.fix_delay


    def calculate_reward(self, custom_func=None, custom_func_args=None):
        '''
        Reward calculation
        :return:
        '''

        if not custom_func:
            if np.abs(self.inventory_level) < self.reward_level:
                return 1.

            else:
                return 0.

        else:
            return custom_func(x=self.inventory_level, **custom_func_args)



    def step(self, action):
        '''
        Function to proceed step in the environment. Behave like OpenAI gym's env.step(action)
        :param action: action to proceed
        :return:
        '''
        self.action = self.action_range[action]
        self.time += 1
        self.demand_last = self.demand_next
        self.demand_next = self.demand_generation_function()
        self.inventory_level -= self.demand_last
        self.upcoming_goods[int(np.round(self.delayed()) + self.time)] += self.action
        self.recieved = self.upcoming_goods[self.time]
        self.inventory_level += self.recieved
        self.upcoming_n_steps = self.upcoming_goods[self.time: self.time + self.next_n_steps]
        self.sum_all_ordered = self.upcoming_goods[self.time:].sum()

        #TODO: add sum of all orders

        self.recieved_goods.append(self.recieved)
        if self.time > self.max_num_steps:
            self.done = True

        return (self.inventory_level, self.demand_next, self.recieved, *self.upcoming_n_steps), \
               self.calculate_reward(custom_func=self.custom_func, custom_func_args=self.custom_func_args), self.done, {}


    def reset(self):
        '''
        Reset the environment on new trajectory
        :return:
        '''
        self.inventory_level = 0
        self.time = 0
        self.ordered_goods = []
        self.recieved_goods = []
        self.upcoming_goods = np.zeros(self.max_num_steps)
        return (self.inventory_level, self.demand_next, *[0 for _ in range(self.next_n_steps)], 0)


    def render(self, close):
        '''
        Renders the environment
        :return: None
        '''
        print('#{}    {} <- |{}| -> ({}) <- {}'.format(self.time, self.demand, self.inventory_level, self.action, self.recieved))
        # print(tabulate(pd.DataFrame(self.upcoming_goods).T, tablefmt='psql'))

        # Color
        R = "\033[0;31;40m"  # RED
        G = "\033[0;32;40m"  # GREEN
        Y = "\033[0;33;40m"  # Yellow
        B = "\033[0;34;40m"  # Blue
        N = "\033[0m"  # Reset
        t = PrettyTable(range(len(self.upcoming_goods)))

        upcoming = list(map(str, self.upcoming_goods.copy()))
        upcoming[self.time] = str(R + str(upcoming[self.time]) + N)

        t.add_row(upcoming)
        print(t)
