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
                 delay_factor=0,
                 delay_mean=2,
                 delay_var=1,
                 min_demand=2,
                 max_demand=30,
                 fix_delay=3,
                 next_n_steps=10,
                 demand_generation_function=None):

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
        self.fix_delay = fix_delay
        self.next_n_steps = next_n_steps
        self.demand_generation_function = demand_generation_function
        self.demand_next = self.demand_generation_function()

        self.action_min = 0
        self.action_min = 10

        self.action_range = list(range(0, 10))

        #configuration
        self.time = -1

        #state variables
        self.ordered_goods = []
        self.recieved_goods = []
        self.upcoming_goods = np.zeros(self.max_num_steps)

        # self.demand = 0
        self.max_capability_of_storage = max_capability_of_storage

        #delay
        self.delay_factor = delay_factor
        self.delay_mean = delay_mean
        self.delay_var = delay_var
        self.d = []
        self.r = []
        self.dif = []
        self.il = []

        self.max_shipment = 5


    def delayed(self):
        # if np.random.binomial(1, self.delay_factor):
        #     return self.fix_delay + np.random.normal(self.delay_mean, self.delay_var)
        # else:
        return 2


    def calculate_reward(self):
        #TODO: change logic

        # return -np.abs(self.inventory_level) * self.c_IL_positive
        #* self.c_IL_positive + (self.inventory_level >= 0) * self.c_IL_negative

        if np.abs(self.inventory_level) < 10:
            return 1

        else:
            return 0
        # elif np.abs(self.inventory_level) < 10:
        #     return 10
        # elif np.abs(self.inventory_level) < 20:
        #     return 5
        # elif np.abs(self.inventory_level) < 30:
        #     return 1
        # else:
        #     return -1




    def step(self, action):
        self.action = self.action_range[action]
        self.time += 1

        self.demand_last = self.demand_next
        self.demand_next = self.demand_generation_function()

        self.inventory_level -= self.demand_last
        self.upcoming_goods[int(np.round(self.delayed()) + self.time)] += self.action

        self.recieved = self.upcoming_goods[self.time]

        # self.dif.append(self.recieved - self.demand)
        # self.d.append(self.demand)
        # self.r.append(self.recieved)
        # self.il.append(self.inventory_level)

        self.inventory_level += self.recieved

        self.upcoming_n_steps = self.upcoming_goods[self.time: self.time + self.next_n_steps]
        self.sum_all_ordered = self.upcoming_goods[self.time:].sum()

        #TODO: add sum of all orders

        self.recieved_goods.append(self.recieved)
        if self.time > self.max_num_steps:
            self.done = True

        return (self.inventory_level, self.demand_next, self.recieved, *self.upcoming_n_steps), self.calculate_reward(), self.done, {}


    def reset(self):
        self.inventory_level = 0
        self.time = 0
        self.ordered_goods = []
        self.recieved_goods = []
        self.upcoming_goods = np.zeros(self.max_num_steps)
        return (self.inventory_level, self.demand_next, *[0 for _ in range(self.next_n_steps)], 0)


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






