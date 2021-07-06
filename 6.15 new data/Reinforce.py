import sys
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable
import pdb


class Policy(nn.Module):
    def __init__(self,  inputs_size, action_size, hidden_size=128):
        super(Policy, self).__init__()
        self.action_size = action_size
        self.linear1 = nn.Linear(inputs_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_size)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        action_scores = self.linear2(x)
        # 输出各个动作的概率
        return F.softmax(action_scores, dim=0)


class REINFORCE:
    def __init__(self, num_inputs, action_space, hidden_size=128):
        self.action_space = action_space
        self.model = Policy(num_inputs, len(action_space), hidden_size)
        self.model.to(dtype=torch.float64)
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()

    def select_action(self, state):
        probs = self.model(Variable(state).cuda())
        # 多项式采样，值越大，被采到的概率越大
        action = probs.multinomial(num_samples=1).data
        prob = probs[action[0]].view(1, -1)
        # 计算logP(r)值
        log_prob = prob.log()
        return action, log_prob, probs

    def update_parameters(self, rewards, log_probs, gamma):
        # rewards, log_probs, entropies 均需要传入数组，一条轨迹上的所有数据
        R = torch.zeros(1, 1).cuda()
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            loss = loss + (log_probs[i]*Variable(R).expand_as(log_probs[i])
                           ).sum()
        loss = loss / len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪防止过拟合，最大L2范数为40
        '''
        clip_grad_norm(parameters: _tensor_or_tensors, max_norm: float, norm_type: float=2., error_if_nonfinite: bool=False) -> torch.Tensor
        对所有参数计算范数（L1 或 L2），当结果大于 max_norm 时，对所有参数都加减相同的值，直到范数在给定的范围之内
        '''
        utils.clip_grad_norm_(self.model.parameters(), 50)
        self.optimizer.step()
        return loss
