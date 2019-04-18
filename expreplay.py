"""expreplay.py: The experience replay module."""
__author__ = "Erik Gärtner"

from collections import deque

import numpy as np


class ExpReplay():
    """
    A generic class for collecting experiences.
    """

    def __init__(self, size):
        """
        Size is the maximum number of collected experiences.
        """
        self.memory = deque(maxlen=size)

    def add(self, exp):
        """
        Add an experience to the queue.
        """
        self.memory.append(exp)

    def sample(self, size):
        """
        Method to sample a collection experiences from the memory.
        Note that this is mostly used in off-policy methods such as Q-learning.
        """
        index = np.random.choice(np.arange(len(self.memory)),
                                 size=size,
                                 replace=False)
        return [self.memory[i] for i in index]

    def get_all(self):
        """Returns all collected experiences"""
        return list(self.memory)

    def clear(self):
        """Empty the memory."""
        self.memory.clear()
