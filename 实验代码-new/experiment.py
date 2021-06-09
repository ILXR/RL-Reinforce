#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Yixuehao
"""
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch

from Reinforce import REINFORCE
from Environment import Environment
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
        s.append(r / (episode + 1))  # average_regret
    return s


# 定义算法的运行环境
'''
main code
'''

armp = DNNPartition(23)

algorithm_random = Random(23)
algorithm_greedy = EpsilonGreedy(23, 0.2)
algorithm_ucb = UCB1(23)


ss_random = np.zeros([1, 600])
ss_greedy = np.zeros([1, 600])
ss_ucb = np.zeros([1, 600])


# for i in range(len(algorithms)):
ss_random = algorithm_run(armp, algorithm_random, 600)
ss_greedy = algorithm_run(armp, algorithm_greedy, 600)
ss_ucb = algorithm_run(armp, algorithm_ucb, 600)


############### REINFORCE ###############
# Params
Hidden_Size = 128
Gamma = 0.99
Episodes = 600  # 训练轮数
Steps = 1  # 每轮的步数


# Env and Reinforce model
env = Environment()
agent = REINFORCE(env.state_size(), env.action_space(), Hidden_Size)

# train
reinforce_rewards = []
for episode in range(Episodes):
    entropies = []
    log_probs = []
    rewards = []
    state = env.random_state()
    for t in range(Steps):
        action, log_prob, entropy = agent.select_action(
            torch.from_numpy(state))
        reward = env.get_reward(state, action)

        entropies.append(entropy)
        log_probs.append(log_prob)
        rewards.append(reward)
        state = env.random_state()
    agent.update_parameters(rewards, log_probs, entropies, Gamma)
    reinforce_rewards.append(np.mean(rewards))


ls_array = ['-', '-.', '--', ':', '-', '-']
marker_array = ['v', 's', '^', 'd', 'o', '.']

# plt.plot(range(600), ss[0, :], label = '$\epsilon$-greedy (0.1)', color='b', ls='--', linewidth=1.5)
plt.plot(range(600), ss_random, label='Random', ls='--',
         color='b', marker='v', markevery=40, linewidth=1.5)
plt.plot(range(600), ss_greedy, label='Greedy', ls='-.',
         color='r', marker='s', markevery=40, linewidth=1.5)
plt.plot(range(600), ss_ucb, label='UCB', ls='-',
         color='g', marker='^', markevery=40, linewidth=1.5)
plt.plot(range(600), reinforce_rewards, label="ReinForce", ls='-',
         color="orange", marker="d", markevery=40, linewidth=1.5)


plt.xlabel('Episode', fontsize=15)
plt.ylabel('Latency', fontsize=15)
plt.legend(loc='best', fontsize=12)
# plt.savefig("C:/Users/YixueHao/Desktop/UCB.jpg")
# plt.clf()
plt.show()
