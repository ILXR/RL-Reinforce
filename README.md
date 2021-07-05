# RL-Reinforce

强化学习-Reinforce 调研与实现

## Log

---

### 2021/7/1

* [x] 调研-优势函数
  * 优势函数其实就是将Q-Value“归一化”到Value baseline上，这样有助于提高学习效率，同时使学习更加稳定；同时经验表明，优势函数也有助于减小方差，而方差过大是导致过拟合的重要因素。
  * 实现方法：使用状态价值V 函数作为Baseline，优势函数 A = Q - V，使用A 重写Loss 中的价值函数

---

### 2021/7/5

* [x] 调研-important sampling
  * 蒙特卡罗采样 --- 蒙特卡洛方法是一种近似推断的方法，通过采样大量粒子的方法来求解期望、均值、面积、积分等问题，蒙特卡洛对某一种分布的采样方法有直接采样、接受拒绝采样与重要性采样三种，直接采样最简单，但是需要已知累积分布的形式。接受拒绝采样与重要性采样适用于原分布未知的情况，这两种方法都是给出一个提议分布，不同的是接受拒绝采样对不满足原分布的粒子予以拒绝，而重要性采样则是给予每个粒子不同的权重
    * 直接采样: 直接采样的方法是根据概率分布进行采样。对一个已知概率密度函数与累积概率密度函数的概率分布，我们可以直接从累积分布函数（cdf）进行采样，在其值域[0, 1]上均匀采样，然后通过cdf的反函数获取x
    * 接受-拒绝采样: p(z)是我们希望采样的分布，q(z)是我们提议的分布(proposal distribution)，令kq(z)>p(z)，我们首先在kq(z)中按照直接采样的方法采样粒子，接下来判断这个粒子落在途中什么区域，对于落在灰色区域的粒子予以拒绝，落在红线下的粒子接受，最终得到符合p(z)的N个粒子
    ![接受-拒绝采样](https://pic1.zhimg.com/80/v2-7d42fbbccd4b7780de8c5b0444846fa4_720w.jpg)
    * 重要性采样(important sampling): 给予每个粒子不同的权重，使用加权平均的方法来计算期望
  * 什么时候会用到重要性采样？
    * 强化学习算法有两个策略：一个用于决策的 *behavior policy μ*，一个用于更新的 *target policy π*，当*μ*和*π*一样时就是On-Policy，不一样时就是Off-Policy算法。
    * 在贝尔曼方程中需要计算的下一个状态的Q函数是基于概率分布*π*的，而Off-Policy使用*μ*进行决策(为了进行探索)，两种策略不一致时才需要重要性采样
