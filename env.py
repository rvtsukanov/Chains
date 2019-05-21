import torch
import numpy as np
import gym

class ChainAgent(gym.Env):
    def __init__(self, inventory_level=0,
                 max_num_steps=1000,
                 max_capability_of_storage=20,
                 delay_factor=0.5,
                 delay_mean=2,
                 delay_var=1,
                 min_demand=2,
                 max_demand=30):

        self.inventory_level = inventory_level
        self.max_num_steps = max_num_steps
        self.done = False
        self.c_IL_positive = 1
        self.c_IL_negative = 1
        self.min_demand = min_demand
        self.max_demand = max_demand

        #configuration
        self.time = 0

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
        if self.inventory_level >= 0:
            return self.inventory_level * self.c_IL_positive
        else:
            return self.inventory_level * self.c_IL_negative


    def step(self, action):
        self.time += 1

        self.demand = np.random.randint(self.min_demand, self.max_demand)
        self.inventory_level -= self.demand


        # print(np.round(self.delayed()) + self.time)

        self.upcoming_goods[int(np.round(self.delayed()) + self.time)] += action
        self.ordered_goods = self.upcoming_goods[self.time:np.max(np.nonzero(self.upcoming_goods))]

        recieved = self.upcoming_goods[self.time]
        self.recieved_goods.append(recieved)
        self.inventory_level += recieved

        if self.time > self.max_num_steps:
            self.done = True

        return (self.inventory_level, self.ordered_goods, recieved, self.demand), self.calculate_reward(), self.done, {}


    def reset(self):
        self.time = 0
        self.ordered_goods = []
        self.recieved_goods = []
        self.upcoming_goods = np.zeros(self.max_num_steps)
        return (self.inventory_level, self.ordered_goods, 0, self.demand)


    def render(self):
        print('upcoming: ', self.upcoming_goods)
        print('oredered: ', self.ordered_goods)
        print('time: ', self.time)




