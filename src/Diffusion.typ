// #set text(font: "Noto Sans SC")
#set heading(numbering: "1.")
#outline()

= Diffusion Model

== 加噪过程

定义一个*马尔可夫链*，从数据 $x_0 ~ q(x)$ 开始，每一步加一点噪声：

#let NN = $cal(N)$
#let LL = $cal(L)$

$
  q(x_t | x_(t-1)) = NN(x_t\; sqrt(1 - beta_t) x_(t-1), beta_t I)
$

使用重参数化，等价于

$
  x_t = sqrt(1 - beta_t) x_(t-1) + sqrt(beta_t) epsilon_t, quad epsilon_t ~ N(0, I),
$

其中 $beta$ 是一系列预定义的参数.

对于两个高斯分布 $x_1 ~ NN(mu_1, sigma_1^2)$ 和 $x_2 ~ NN(mu_2, sigma_2^2)$，我们有：

$
  x_1 + x_2 ~ NN(mu_1 + mu_2, sigma_1^2 + sigma_2^2)
$

于是定义 $alpha_t = 1 - beta_t$，则

$
  x_t &= sqrt(alpha_t) x_(t-1) + sqrt(1 - alpha_t) epsilon_t \
      &= sqrt(alpha_t alpha_(t-1)) x_(t-2) + sqrt(alpha_t (1 - alpha_(t-1))) epsilon_t + sqrt(1 - alpha_t) epsilon_t \
      &= sqrt(alpha_t alpha_(t-1)) x_(t-2) + sqrt(1 - alpha_t alpha_(t-1)) epsilon \
      &= sqrt(alpha_t alpha_(t-1) alpha_(t-2)) x_(t-3) + sqrt(1 - alpha_t alpha_(t-1) alpha_(t-2)) epsilon \
      &= dots
$

定义 $overline(alpha)_t = product_(s=1)^t alpha_s$, 得到 $x_t$ 的封闭形式

$
  x_t = sqrt(overline(alpha)_t) x_0 + sqrt(1 - overline(alpha)_t) epsilon, quad epsilon ~ N(0, I).
$

即

$
  q(x_t | x_0) = NN(x_t\; sqrt(overline(alpha)_t) x_0, (1 - overline(alpha)_t) I)
$

因为 $alpha_t < 1$, 因此 $overline(alpha)_infinity -> 0$，此时 $q(x_t | x_0) -> NN(0, I)$. 
即随着 $t$ 增大，$x_t$ 趋近于纯噪声.

== 去噪过程

我们的目标是找到加噪过程 $q(x_t | x_(t-1))$ 的逆过程，即逐步去噪的条件高斯分布

$
  q(x_(t-1) | x_t) = NN(x_(t-1)\; mu_t (x_t), Sigma_t (x_t))
$

$q(x_(t-1) | x_t)$不可解，因此使用一个模型 $theta$ 近似，以 $x_t$ 和 $t$ 作为输入，输出分布参数

$
  p_theta (x_(t-1) | x_t) = NN(x_(t-1)\; mu_theta (x_t, t), Sigma_theta (x_t, t))
$

== 最大似然估计 MLE

目标是最小化去噪模型 $p_theta$ 对初始数据 $x_0$ 的负对数似然：

$
  LL(theta) = - log p_theta (x_0)
$

同样地，使用变分下界 ELBO 优化，其中 $x_0$ 是已知变量，$x_(1:T)$ 是隐变量：

$
 cal(F)(q, theta) = EE_q(x_(1:T)|x_0) [log p_theta (x_(0:T)) - log q(x_(1:T)|x_0)] \
 LL(theta) <= - cal(F)(q, theta)
$

根据马尔可夫性质，

$
  q(x_(1:T)|x_0) = q(x_T | x_0) product_(t=2)^T q(x_(t-1) | x_t, x_0) \
  p_theta (x_(0:T)) = p_theta (x_T) product_(t=1)^T p_theta (x_(t-1) | x_t)
$

代入得到 $cal(F)(q, theta)$

$
   =& EE_q(x_1:x_T|x_0) [log p_theta (x_T) + sum_(t=1)^T log p_theta (x_(t-1) | x_t) - log q(x_T | x_0) - sum_(t=2)^T q(x_(t-1) | x_t,x_0)] \

  =& EE_q(x_1|x_0) [log p_theta (x_0|x_1)]
  - sum_(t=2)^T EE_q(x_t,x_(t-1)|x_0) [log q(x_(t-1) | x_t,x_0) - log p_theta (x_(t-1) | x_t)] \
  &+ EE_q(x_T|x_0) [log p_theta (x_T) - log q(x_T | x_0)] \

  =& EE_q(x_1|x_0) [log p_theta (x_0|x_1)]
  - sum_(t=2)^T EE_q(x_t|x_0) EE_q(x_(t-1)|x_t,x_0) [log q(x_(t-1) | x_t,x_0) - log p_theta (x_(t-1) | x_t)] \
  &+ EE_q(x_T|x_0) [log p_theta (x_T) - log q(x_T | x_0)] \

  =& EE_q(x_1|x_0) [log p_theta (x_0|x_1)]
  - sum_(t=2)^T EE_q(x_t|x_0) "KL"(q(x_(t-1) | x_t,x_0) || p_theta (x_(t-1) | x_t)) \
  &- "KL"(log q(x_T | x_0) || log p_theta (x_T)) \
$

分别记作

$
  LL := -cal(F)(q, theta) = LL_0 + sum_(t=2)^(T) LL_(t-1) + LL_T.
$

其中

- 初始项 $LL_0 = -EE_q(x_1|x_0) [log p_theta (x_0|x_1)]$
  - 最小化最后一步 $x_1$ 到 $x_0$ 的重建误差
  - 当 $p_theta$ 取高斯分布时，相当于最小化均方误差 MSE: $EE[norm(x_1 - x_0)^2]$
- 中间项 $LL_(t-1) = EE_q(x_t|x_0) "KL"(q(x_(t-1) | x_t,x_0) || p_theta (x_(t-1) | x_t))$ 
  - 让模型 $p_theta$ 学习去噪过程，缩小和真实后验 $q(x_(t-1) | x_t,x_0)$ 的差距。
  - $ q(x_(t-1)|x_t,x_0) = (q(x_(t-1)|x_0) q(x_t|x_(t-1))) / q(x_t|x_0) prop q(x_(t-1)|x_0) q(x_t|x_(t-1)) $
- 终端项 $LL_T = "KL"(log q(x_T | x_0) || log p_theta (x_T))$
  - 让模型 $p_theta$ 的先验分布接近 $q(x_T | x_0)$
  - 对于足够大的 $T$，$q(x_T | x_0) approx NN(0, I)$，而先验 $p_theta (x_T)$ 被定义为 $NN(0, I)$
  - 因此该项可以*忽略不计*.

= Score-based Diffusion Model & Langevin dynamics

==  Langevin dynamics

定义 score 函数，即数据分布的对数梯度：

$
  "score"(x) = nabla_x log p(x)
$

Langevin dynamics 指出，如果已知一个分布 $pi(x)$ 的 score，则可以通过以下迭代从该分布（离散）采样：

$
  X_(t+tau) = X_t + tau nabla_x log pi(X_t) + sqrt(2 tau) xi, quad xi ~ N(0, I)
$

Langevin dynamics 描述的是粒子在势能场中的布朗运动，是一种带随机扰动的梯度上升过程，既偏向于高概率区域，也保证了采样的随机性.

在扩散模型中，这就是反向去噪采样的核心形式。

== Tweedie 定理

Tweedie 定理指出，对于 $x ~ p(x) = cal(N)(x; mu, Sigma)$，

$
  EE[mu|x] = x + Sigma nabla_x log p(x)
$

即使用数据 $x$ 和 score 函数，可以估计出原始均值 $mu$.

== 去噪问题

考虑加噪过程

$
  y = x + n; quad n ~ NN(0, sigma^2 I)
$

的逆过程，即从 $y$ 估计 $hat(x)$.

- Bayes 观点（MLE）： $hat(x) = arg max_(x) log p(x|y)$
- Score 观点（Langevin dynamics）： 在 Langevin dynamics 中减小噪声项，对 $p(y)$ 取样，得到的就是 $arg max_y p(y) = x$. 如果解码器预测 score $nabla_y log p(y)$，则可以通过 Langevin dynamics 采样得到 $x$ 的估计，其中一次迭代即对应一次去噪过程。
  - 根据 Tweedie 定理，预测 score 隐式包含了一个去噪器 $EE[x|y]$.
  - 如果步长 $tau$ 递减，则类似于模拟退火，最终收敛到 $p(y)$ 的极大值点，即 $x$ 的估计.

= 条件 Diffusion Model

给定条件 $c$, 需要

$
  nabla_x log p(x|c)
$

Bayes 公式：

$
  nabla_x log p(x|c) = nabla_x log p(x) + nabla_x log p(c|x)
$

如何得到 $nabla_x log p(c|x)$:

classifier guidance：基于 $x_t$ 的分类器，通过反向传播获取梯度。
通过给 $nabla_x log p(c|x)$ 乘以一个系数 $s > 1$，可以增强条件信息的影响，得到更符合条件的样本生成结果.

= Classifier-Free Diffusion Guidance (CFG)

不使用分类器，而是定义两种去噪模型的score：

- 条件模型 $nabla_x log p_theta (x_t, c) = 1/sigma^2 (D_theta (x_t, sigma, c) - x_0)$
- 无条件模型 $nabla_x log p_theta (x_t) = 1/sigma^2 (D_theta (x_t, sigma) - x_0)$

其中 $D_theta$ 是去噪器, 使用一个 null 条件（如全零向量）来表示无条件模型. 则上式的等价形式

// 使用二者的凸组合来进行采样：

$
  nabla_x log p(x|c) &= nabla_x log p(x) + S nabla_x log p(c|x) \
  &= nabla_x log p_theta (x_t) + S (nabla_x log p_theta (x_t, c) - nabla_x log p_theta (x_t)) \
  &= S nabla_x log p_theta (x_t, c) + (1 - S) nabla_x log p_theta (x_t) \
  &= 1/sigma^2 (S D_theta (x_t, sigma, c) + (1 - S) D_theta (x_t, sigma) - x_0)
$

即用二者的凸组合来进行去噪采样.