import matplotlib.pyplot as plt
import numpy as np
import torch

from Reinforce import REINFORCE
from Environment import Environment
from Armdata import DNNPartition
from Algorithm import EpsilonGreedy, UCB1, Random

Episodes = 1000  # 训练轮数
L2_reg = True
Clip_norm = False


def algorithm_run(armp, algorithm, episodes):
    r, s = 0.0, []
    for episode in range(episodes):
        arm = algorithm.pull()
        reward = armp.step(arm)
        # print('Episode %s: arm = %s , reward = %.1f' % (episode, arm, reward))
        algorithm.update(arm, reward)
        r += reward
        s.append(r / (episode + 1))  # average_regret
    return s


'''
main code
'''
armp = DNNPartition(23)

algorithm_random = Random(23)
algorithm_greedy = EpsilonGreedy(23, 0.2)
algorithm_ucb = UCB1(23)

ss_random = np.zeros([1, Episodes])
ss_greedy = np.zeros([1, Episodes])
ss_ucb = np.zeros([1, Episodes])

# for i in range(len(algorithms)):
ss_random = algorithm_run(armp, algorithm_random, Episodes)
ss_greedy = algorithm_run(armp, algorithm_greedy, Episodes)
ss_ucb = algorithm_run(armp, algorithm_ucb, Episodes)


############### REINFORCE ###############
# Params
Hidden_Size = 128
Gamma = 0.99  # Discount rate
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
agent_baseline = REINFORCE(env.state_size(), env.action_space(), Hidden_Size)

# train
reinforce_rewards = []
reinforce_loss = []
print("REINFORCE")
for episode in range(Episodes):
    log_probs = []
    entropies = []
    rewards = []
    for t in range(Steps):
        state = env.random_state()
        action, log_prob, probs, entrop = agent.select_action(
            torch.from_numpy(state))
        reward = env.get_reward(state, action)
        rewards.append(reward)
        entropies.append(entrop)
        log_probs.append(log_prob)
    loss = agent.update_parameters(
        rewards, log_probs, entropies, Gamma, L2_reg, Clip_norm)
    mean_reward = np.mean(rewards)
    reinforce_rewards.append(mean_reward)
    reinforce_loss.append(loss)
    if(episode % 10 == 0):
        print(f"Episode {episode}: loss = {loss}")
reinforce_rewards = rolling_mean(reinforce_rewards, size=15)

reinforce_baseline_rewards = []
reinforce_baseline_loss = []
print("\nREINFORCE with advantage function")
for episode in range(Episodes):
    log_probs = []
    entropies = []
    rewards = []
    rewards_baseline = []
    for t in range(Steps):
        state = env.random_state()
        best_action, best_reward = env.best_step(state)

        action, log_prob, probs, entrop = agent_baseline.select_action(
            torch.from_numpy(state))

        reward = env.get_reward(state, action)
        rewards.append(reward)
        entropies.append(entrop)

        state_rewards = torch.from_numpy(env.get_rewards(state)).cuda()
        adv_func = (probs.mul(state_rewards)).sum()
        rewards_baseline.append(reward-adv_func)

        log_probs.append(log_prob)
    loss = agent_baseline.update_parameters(
        rewards_baseline, log_probs, entropies, Gamma, L2_reg, Clip_norm)
    mean_reward = np.mean(rewards)
    reinforce_baseline_loss.append(loss)
    reinforce_baseline_rewards.append(mean_reward)
    if(episode % 10 == 0):
        print(f"Episode {episode}: loss = {loss}")
reinforce_baseline_rewards = rolling_mean(reinforce_baseline_rewards, size=15)

# Draw Curves
x_list = range(Episodes)
ls_array = ['-', '-.', '--', ':', '-', '-']
marker_array = ['v', 's', '^', 'd', 'o', '.']

plt.figure()
title = f'Clip norm: {Clip_norm}\nL2 Regularization: {L2_reg}'
plt.title(title)
plt.plot(x_list, ss_random, label='Random', ls='--',
         color='b', marker='v', markevery=40, linewidth=1.5)
plt.plot(x_list, ss_greedy, label='Greedy', ls='-.',
         color='r', marker='s', markevery=40, linewidth=1.5)
plt.plot(x_list, ss_ucb, label='UCB', ls='-',
         color='g', marker='^', markevery=40, linewidth=1.5)
plt.plot(x_list, reinforce_rewards, label="ReinForce", ls='-',
         color="orange", marker="d", markevery=40, linewidth=1.5)
plt.plot(x_list, reinforce_baseline_rewards, label="ReinForce-advantage function", ls='-',
         color="slateblue", marker="o", markevery=40, linewidth=1.5)
plt.xlabel('Episode', fontsize=15)
plt.ylabel('Latency', fontsize=15)
plt.legend(loc='best', fontsize=12)
plt.ylim(ymax=15000, ymin=1000)
manager = plt.get_current_fig_manager()
manager.window.state('zoomed')

file_name = title.replace("\n", " -- ").replace(": ", "-")
# plt.savefig("6.15 new data/result/"+"reward -- "+file_name+".png")

plt.figure()
plt.title(title)
plt.plot(x_list, reinforce_loss,  label="ReinForce", ls='-',
         color="orange", marker="d", markevery=40, linewidth=1.5)
plt.plot(x_list, reinforce_baseline_loss, label="ReinForce-advantage function", ls='-',
         color="slateblue", marker="o", markevery=40, linewidth=1.5)
plt.xlabel('Episode', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.legend(loc='best', fontsize=12)
manager = plt.get_current_fig_manager()
manager.window.state('zoomed')
# plt.savefig("6.15 new data/result/"+"loss -- "+file_name+".png")

plt.show()
