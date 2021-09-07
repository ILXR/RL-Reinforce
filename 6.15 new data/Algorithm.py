#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Yixuehao
"""
# -*- coding: utf-8 -*-


import numpy as np

# 此处对本文用到的三个算法进行分析


class EpsilonGreedy():
    # 定义epsilongreedy算法
    def __init__(self, arm_k, epsilon):
        self.arm_k = arm_k
        self.epsilon = epsilon
        self.values = np.zeros(self.arm_k)
        self.counts = np.zeros(self.arm_k)

    def pull(self):
        for arm in range(self.arm_k):
            if self.counts[arm] == 0:
                return arm
        greedyindex = np.random.random()
        # print('greedyindex:',greedyindex)
        if greedyindex < self.epsilon:
            return np.random.randint(0, self.arm_k)
        else:
            return np.argmin(self.values)
            # return np.argsort(-self.values)[0]

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        # print(self.values)

# 定义UCB算法类


class UCB1():
    def __init__(self, arm_k):
        self.arm_k = arm_k
        self.values = np.zeros(self.arm_k)
        self.counts = np.zeros(self.arm_k)
        self.UCB = np.zeros(self.arm_k)

    def pull(self):
        for arm in range(self.arm_k):
            if self.counts[arm] == 0:
                return arm
        t = np.sum(self.counts)
        for arm in range(self.arm_k):
            self.UCB[arm] = self.values[arm] - \
                np.sqrt(2 * np.log(t) / self.counts[arm])
        # print(self.UCB)
        return np.argmin(self.UCB)
        # 此处可以选取多个臂
        # return np.argsort(-self.UCB)[0：m]

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


# 定义Randoms算法
class Random():
    def __init__(self, arm_k):
        self.arm_k = arm_k
        self.values = np.zeros(self.arm_k)
        self.counts = np.zeros(self.arm_k)

    def pull(self):
        for arm in range(self.arm_k):
            if self.counts[arm] == 0:
                return arm

        return np.random.randint(0, self.arm_k)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
