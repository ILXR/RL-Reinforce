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
Gamma = 0.99  # Discount rate
Episodes = 600  # 训练轮数
Steps = 4  # 每轮的步数
'''Note
当估计action的值（reward）的时候，强化学习算法一般把这个action导致的所有reward进行求和
求和的时候给immediate rewards（【即时奖励，时间上较近的奖励】）更大的权重，
later rewards（【时间上较远的奖励】）更小的权重（认为一个动作对未来一小段时间的影响大于未来一大段时间的影响）。
为了对这种情况进行建模，每个时间步应用一个discount rate
如果你觉得未来重要，为了最后获得reward，你可能会近期承受很多pain；
然而如果你觉得未来不重要，你可能会更看重 immediate rewards（【眼前利益】），而不是未来的投资。
'''


def rolling_mean(data, size=10):
    result = []
    length = len(data)
    half = int(size/2)
    for i in range(length):
        start = max(0, i-half+1)
        end = min(i+size-half, length)
        tem = data[start: end]
        result.append(np.mean(tem))
    return result


def smooth_average(record, n):
    # 对结果进行平均处理
    res = []
    length = len(record)
    for i in range(int(length/n)):
        average = sum(record[n*i:n*(i+1)]) / n
        res.append(average)
    return res


# Env and Reinforce model
env = Environment()
agent = REINFORCE(env.state_size(), env.action_space(), Hidden_Size)

# train
reinforce_rewards = []
for episode in range(Episodes):
    entropies = []
    log_probs = []
    rewards = []
    for t in range(Steps):
        state = env.random_state()
        action, log_prob, entropy = agent.select_action(
            torch.from_numpy(state))
        reward = env.get_reward(state, action)

        entropies.append(entropy)
        log_probs.append(log_prob)
        rewards.append(reward)
    loss = agent.update_parameters(rewards, log_probs, entropies, Gamma)
    mean_reward = np.mean(rewards)
    reinforce_rewards.append(mean_reward)
    if(episode % 10 == 0):
        print(f"Episode {episode}: loss = {loss}")
#         print(
#             f"Episode {episode}: state = {state} , arm = {action[0]} , reward_mean = {mean_reward}")
reinforce_rewards = rolling_mean(reinforce_rewards, size=15)



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
