+++
title = '如何利用 LLM 辅助学习：以 DQN 的习得为例'
date = 2025-02-04T20:02:17-08:00
draft = true
math = true
+++
> ✨
> 在 ChatGPT 出现之后，我找到了一条非常适合自学新知识的路径。这结合了我本科时学习专业知识的方法，也充分利用了 ChatGPT 作为我的 RLCF (reinforcement learning from chatgpt feedback 🫶)。最近在学习强化学习相关的知识，于是把我的学习路径记录下来。

对计算机科学领域的知识而言，往往我们都是在现有的基础上去解决问题。比如大家在计算机系统课程中学习 Cache 相关的知识，就需要搞明白两件事：

1. 前人要通过 Cache 解决的问题是什么？
2. Cache 是如何解决这个问题的？

实际上，我一直认为第一个问题远比第二个问题重要。下面我将以自己学习 DQN 为例。

# Step1: 利用手稿找到核心问题
对于一个我并不熟悉的领域，阅读一份博客或者综述论文是一个很好的开始。综述论文严谨完整，但是耗费的时间相对较长，所以我更推荐从博客开始。不过需要注意的是，网络上的博客文章质量良莠不齐，甚至经常会出现错误，最好是和其他资料交叉验证。

学习 DQN，我参考的资料是，
[动手学强化学习 第七章 DQN 算法](https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95/)。这份资料在 Google Search 的前几名，一般质量还是有保障的。

在阅读概览材料的时候，就一定要把握住 **DQN 解决了什么问题**？DeepMind 当初设计 DQN，一定是遇到了一些传统强化学习和传统监督学习解决不了的事情，否则没必要设计 DQN 出来。

我对强化学习的知识近乎于 0，因此需要先花时间了解为什么有强化学习然后才是 Q-Learning 和 DQN。带着 Q-learning 甚至是Reinforcement Learning 的 motivation 是什么这个问题开始阅读，这个过程一定要和之前的基础进行对比。

例如，了解强化学习的motivation 就一定要知道有监督学习为什么不够用。

$$
f(x;\theta)^* = \argmin_{\theta} E_{x \sim \text{Data}} \text{Loss}(y, x) \tag 1
$$

公式（1）是有监督学习的目标，其目的是最小化预测值和真实值的差异。从这个角度看，强化学习强调从与环境的交互中获得反馈的思想并不和有监督学习互斥。**那为什么我们还需要专门提出强化学习呢？**

《动手学强化学习》指出，这是因为一般的监督学习只是在现有的分布里采样，而不和环境主动交互（比如文本分类，并没有策略、交互这些概念，只是关心文本与当前分布）。

$$
\pi(a|s;\theta)^* = \argmin_{\theta} (-E_{\tau\sim\pi}[\sum_{t=0}^T\gamma^t R(s_t,a_t)]) \tag 2
$$

这是强化学习的目标。其中：
- $\pi$ 是(state, action)序列
- $R(s_t,a_t)$是在状态$s_t$下执行动作$a_t$的即时奖励
- $\gamma$ 是折扣因子，用来平衡当前奖励和未来奖励的重要性
- $E_{\tau\sim\pi}$是对策略的一个采样

从实质上，强化学习和监督学习一样，都是要最小化某个函数，不过强化学习要最小化的是 -Reward 函数。
> ✨
> 这里为什么可以让损失函数为 Reward 函数在策略轨迹上的和？

不同强化学习算法就是去设计不同的 Reward 函数，例如 Q-Learning，认为 Reward 函数是 $Q(state, action)$，最优的 Q 可以被计算为：

$$
Q(s_t,a_t) \gets Q(s_t, a_t) + \alpha [R_t + \gamma max_a Q(S_{t+1},a) - Q(s_t, a_t)]
$$

而 DQN 则是用神经网络去学到 Q 函数。