#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Yixuehao
"""
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from Armdata import DNNPartition
from Algorithm import EpsilonGreedy, UCB1, Random


# 运行算法 episode 是运行多少次
def algorithm_run(armp, algorithm, episodes):
    r, s = 0.0, []
    for episode in range(episodes):
        arm = algorithm.pull()
        reward = armp.step(arm)
        print('Episode %s: arm = %s , reward = %.1f' % (episode, arm, reward))
        algorithm.update(arm, reward)
        r += reward
        s.append(r / (episode + 1)) # average_regret
    return s


# 定义算法的运行环境
'''
main code
'''

armp = DNNPartition(23)

algorithm_random = Random(23)
algorithm_greedy = EpsilonGreedy(23, 0.2)
algorithm_ucb = UCB1(23)



ss_random = np.zeros([1,600])
ss_greedy = np.zeros([1,600])
ss_ucb = np.zeros([1,600])



# for i in range(len(algorithms)):
ss_random = algorithm_run(armp, algorithm_random, 600)
ss_greedy = algorithm_run(armp, algorithm_greedy, 600)
ss_ucb = algorithm_run(armp, algorithm_ucb, 600)


ls_array = ['-', '-.', '--', ':', '-', '-']
marker_array = ['v', 's', '^', 'd', 'o', '.']

# plt.plot(range(600), ss[0, :], label = '$\epsilon$-greedy (0.1)', color='b', ls='--', linewidth=1.5)
plt.plot(range(600), ss_random, label = 'Random', ls='--', color ='b', marker='v', markevery=40, linewidth=1.5)
plt.plot(range(600), ss_greedy, label = 'Greedy', ls='-.', color ='r', marker='s', markevery=40, linewidth=1.5)
plt.plot(range(600), ss_ucb, label = 'UCB', ls='-',  color ='g', marker='^', markevery=40, linewidth=1.5)


plt.xlabel('Episode', fontsize=15)
plt.ylabel('Latency', fontsize=15)
plt.legend(loc='best', fontsize=12)
# plt.savefig("C:/Users/YixueHao/Desktop/UCB.jpg")
# plt.clf()
plt.show()
