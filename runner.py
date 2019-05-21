import numpy as np

from env import ChainAgent
agent = ChainAgent()



for i in range(10):
    s = agent.step(np.random.randint(0, 50))
    agent.render()
