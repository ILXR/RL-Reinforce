from Reinforce import REINFORCE
from Environment import Environment
import matplotlib.pyplot as plt
import numpy as np
import torch

# Params
Hidden_Size = 128
Gamma = 0.99
Episodes = 600 # 训练轮数
Steps = 10 # 每轮的步数


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

plt.figure()
plt.title("ReinForce Rewards")
plt.plot(reinforce_rewards, color="blue")
plt.show()
