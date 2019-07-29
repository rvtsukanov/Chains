from collections import namedtuple
import random
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

Trajectory = namedtuple('Trajectory',
                        ('state', 'action', 'reward'))

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class ReplayMemory(object):
    '''
    Class for storing agent's experience
    '''
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
        '''
        Returns a saple of len=batch_size
        :param batch_size: size of batch
        :return:
        '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

