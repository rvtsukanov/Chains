# import torch
import numpy as np
import gym
import pandas as pd
from tabulate import tabulate
from prettytable import PrettyTable

class ChainAgent(gym.Env):
    def __init__(self, inventory_level=0,
                 max_num_steps=15,
                 max_capability_of_storage=20,
                 delay_factor=0.5,
                 delay_mean=5,
                 delay_var=1,
                 min_demand=2,
                 max_demand=30):

        print('=========================================')
        print('INITIAL CONFIGURATION: inventory level {}'.format(inventory_level))
        print('=========================================')

        self.inventory_level = inventory_level
        self.max_num_steps = max_num_steps
        self.done = False
        self.c_IL_positive = 1
        self.c_IL_negative = 1
        self.min_demand = min_demand
        self.max_demand = max_demand

        #configuration
        self.time = -1

        #state variables
        self.ordered_goods = []
        self.recieved_goods = []
        self.upcoming_goods = np.zeros(self.max_num_steps)

        self.demand = 0
        self.max_capability_of_storage = max_capability_of_storage

        #delay
        self.delay_factor = delay_factor
        self.delay_mean = delay_mean
        self.delay_var = delay_var

        self.max_shipment = 5


    def delayed(self):
        if np.random.binomial(1, self.delay_factor):
            return np.random.normal(self.delay_mean, self.delay_var)
        else:
            return 0


    def calculate_reward(self):
        return (self.inventory_level >= 0) * self.c_IL_positive + (self.inventory_level >= 0) * self.c_IL_negative


    def step(self, action):
        self.action = action
        self.time += 1

        self.demand = np.random.randint(self.min_demand, self.max_demand)
        self.inventory_level -= self.demand

        self.upcoming_goods[int(np.round(self.delayed()) + self.time)] += action
        self.ordered_goods = self.upcoming_goods[self.time:np.max(np.nonzero(self.upcoming_goods))]

        self.recieved = self.upcoming_goods[self.time]
        self.recieved_goods.append(self.recieved)
        self.inventory_level += self.recieved

        if self.time > self.max_num_steps:
            self.done = True

        return (self.inventory_level, self.ordered_goods, self.recieved, self.demand), self.calculate_reward(), self.done, {}


    def reset(self):
        self.time = 0
        self.ordered_goods = []
        self.recieved_goods = []
        self.upcoming_goods = np.zeros(self.max_num_steps)
        return (self.inventory_level, self.ordered_goods, 0, self.demand)


    def render(self, close):
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






