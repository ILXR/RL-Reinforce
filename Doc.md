# 说明文档

---

## 代码相关

### 交叉熵

用来衡量在给定的真实分布下，使用非真实分布所指定的策略消除系统的不确定性所需要付出的努力的大小

![交叉熵](./pics/equation.svg)

其中，p 表示真实分布(目标策略), q 表示非真实分布(采取的策略)。在机器学习中的分类算法中，我们总是最小化交叉熵，因为交叉熵越低，就证明由算法所产生的策略最接近最优策略，也间接证明我们算法所算出的非真实分布越接近真实分布。

在代码中，相关部分是这样实现的:

~~~python
entropy = - (probs*probs.log()).sum()
~~~

这个公式是信息熵的计算方法。

> 信息熵 - 信息熵代表的是随机变量或整个系统的不确定性，熵越大，随机变量或系统的不确定性就越大。
>
> ![信息熵](pics/信息熵.svg)

也就是说，在代码中加入了一项信息熵，而不是交叉熵，而且该信息熵在REINFORCE的原始公式中应该是没有对应项的。

---

## 结果分析

### 加入L2范数

~~~python
weight_p, bias_p = [], []
for name, p in self.model.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]
lambd = torch.tensor(1.).cuda()
l2_reg = torch.tensor(0.).cuda()
for param in weight_p:
    l2_reg += torch.norm(param)
    loss += lambd * l2_reg
~~~

只对模型参数中的 weight 进行了处理，加入正则项，结果如下

#### 情况一

- [x] L2正则项
- [x] 剪枝处理
![True-True-Reward](./6.15%20new%20data/result/reward%20--%20Clip%20norm-True%20--%20L2%20Regularization-True.png)
![True-True-Loss](./6.15%20new%20data/result/loss%20--%20Clip%20norm-True%20--%20L2%20Regularization-True.png)

#### 情况二

- [x] L2正则项
- [ ] 剪枝处理
![True-False-Reward](./6.15%20new%20data/result/reward%20--%20Clip%20norm-False%20--%20L2%20Regularization-True.png)
![True-False-Loss](./6.15%20new%20data/result/loss%20--%20Clip%20norm-False%20--%20L2%20Regularization-True.png)

#### 情况三

- [ ] L2正则项
- [ ] 剪枝处理
![False-False-Reward](./6.15%20new%20data/result/reward%20--%20Clip%20norm-False%20--%20L2%20Regularization-False.png)
![False-False-Loss](./6.15%20new%20data/result/loss%20--%20Clip%20norm-False%20--%20L2%20Regularization-False.png)

#### 情况四

- [ ] L2正则项
- [x] 剪枝处理
![False-True-Reward](./6.15%20new%20data/result/reward%20--%20Clip%20norm-True%20--%20L2%20Regularization-False.png)
![False-True-Loss](./6.15%20new%20data/result/loss%20--%20Clip%20norm-True%20--%20L2%20Regularization-False.png)

#### 总结

**Reward 曲线：**

| Reward | 剪枝策略 | L2正则化 |
|:-:|:-:|:-:|
|REINFORCE|有负作用，最终Reward比较差|有减小波动幅度的作用|
|REINFORCE-Baseline|加快了收敛速度，更快的稳定在最优值|整体没有明显影响，加入正则项后在前期反而加大了波动幅度|

**Loss 曲线：**

| Loss | 剪枝策略 | L2正则化 |
|:-:|:-:|:-:|
|REINFORCE|改变了收敛值，加大了曲线波动，可能是欠拟合了|没有明显影响|
|REINFORCE-Baseline|收敛速度大大加快|略微减小收敛速度，加大了波动幅度|

对REINFORCE，最优应该是加入正则项，不采用剪枝

对REINFORCE-Baseline，最优策略应该是加入剪枝策略，正则项加不加都可
