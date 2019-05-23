import numpy as np

from env import ChainAgent
agent = ChainAgent(inventory_level=10)



for i in range(9):
    s = agent.step(np.random.randint(0, 50))
    agent.render(...)
