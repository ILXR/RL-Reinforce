import pandas as pd
import numpy as np
import random


class Environment():
    def __init__(self):
        super().__init__()
        with open("./实验代码-new/DNN_Partition.csv", "r")as f:
            self.array_data = np.loadtxt(f, delimiter=",")
        self.actions = np.arange(0, 23)
        states = []
        rewards = []
        for data in self.array_data:
            states.append([data[2], data[3]])
            rewards.append(data[6])
        self.states = np.array(states)
        self.rewards = np.array(rewards)

    def get_reward(self, state, action):
        for i in range(8):
            index = i*23+action
            if((self.states[index] == state).all()):
                return self.rewards[index]
        return -1

    def random_step(self):
        index = random.randint(0, len(self.rewards))
        reward = self.rewards[index]
        action = index % 22
        state = self.states[index]
        return state, action, reward

    def random_state(self):
        index = random.randint(0, len(self.states)-1)
        return self.states[index]

    def action_size(self):
        return len(self.actions)

    def state_size(self):
        return len(self.states[0])

    def action_space(self):
        return self.actions