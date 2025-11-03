// #set text(font: "Noto Sans SC")
#set heading(numbering: "1.")
#outline()

#let NN = $cal(N)$

= 变分下界 ELBO

在最大似然估计（Maximum Likelihood Estimation, MLE）中，目标是最大化观测数据 $x$ 的对数似然.

定义参数 $theta$ 的对数似然（log likelihood）

$
  ell(theta) = log p_theta (x)
  = log sum_z p_theta (x, z),
$

其中 $x$ 是观测变量，$z$ 是潜在变量.根据 Jensen 不等式，对于任意分布 $q(z)$，有

$
  ell(theta) = log sum_z p_theta (x,z)
  = log sum_z q(z) (p_theta (x,z)) / q(z)
  = log EE_(z~q)[(p_theta (x,z)) / q(z)] \
  >= EE_(z~q) [log (p_theta (x,z)) / q(z)]
  = EE_(z~q) [log p_theta (x,z)] - EE_(z~q) [log q(z)]
$

由此定义 $ell(theta)$ 的变分下界（Evidence Lower BOund, ELBO）

$
  ell(theta) >=
  cal(F)(q, theta) equiv EE_(z~q) [log p_theta (x,z)] - EE_(z~q) [log q(z)].
$

Jensen 不等式的取等条件为

$
  forall z, (p_theta (x,z)) / q(z) = "constant" c
$

因此

$
  p_theta (x) = sum_z p_theta (x,z) = c sum_z q(z) = c \
  q(z) = (p_theta (x,z)) / (p_theta (x)) = p_theta (z|x).
$

这里的 $q(z)$ 可以是任意分布，但使用中常用 $q(z) = q_phi (z|x)$，即 $z$ 的近似后验分布，而近似后验分布越接近真实后验分布 $p_theta (z|x)$，ELBO 越接近对数似然.
= KL 散度

分布 $p(x)$ 的信息熵定义为

$
  H(p) = EE_(x~p)[- log p(x)]
$

即编码长度的期望值.如果认为 $x$ 服从另一分布 $q(x)$，在“真实”分布 $p(x)$ 下的编码长度期望值定义为交叉熵：

$
  H(p, q) = EE_(x~p)[- log q(x)] >= H(p).
$

KL 散度即定义为额外的编码长度

$
  "KL"(p || q) = H(p, q) - H(p) = EE_(x~p)[log p(x) - log q(x)].
$

KL 散度描述了两个分布的差异程度，但是不是距离度量，因为不满足对称性和三角不等式.

== KL 散度的非负性

对于任意分布 $p(x), q(x)$，有

$
      "KL"(p || q) & = EE_(x~p)[log p(x) - log q(x)] \
                   & = -EE_(x~p)[log q(x) / p(x)] \
  ("Jensen") & >= -log EE_(x~p)[q(x) / p(x)] \
                   & = -log integral_x p(x) (q(x) / p(x)) dif x \
                   & = -log integral_x q(x) dif x \
                   & = -log 1 = 0.
$

易知当且仅当 $p = q$ 时，$"KL"(p || q) = 0$.

== ELBO 等价形式

ELBO 的误差为

$
  ell(theta) - cal(F)(q, theta)\
  = EE_(z~q)[log p_theta (x) - log p_theta (x,z) + log q(z)] \
  = EE_(z~q)[-log p_theta (z|x) + log q(z)] \
  = "KL"(q(z) || p_theta (z|x)) >= 0.
$

因此，ELBO 可以写作

$
  cal(F)(q, theta) = ell(theta) - "KL"(q(z) || p_theta (z|x))
$

取等条件同样为 $q(z) = p_theta (z|x)$.

上述形式描述了ELBO和真实对数似然的差异，但 依然有 $ell(theta)$ 项，不便于优化.另一种等价形式为

$
  EE_(z~q) [log p_theta (x,z)] - EE_(z~q) [log q(z)] \
  = EE_(z~q) [log (p_theta (x,z)) / (p_theta (z))] - (EE_(z~q) [log q(z)] + EE_(z~q) [log p_theta (z)]) \
  = EE_(z~q) [log p_theta (x|z)] - "KL"(q(z) || p_theta (z))
$

这是常用的可以直接优化的形式.

= EM

EM 算法的目标是最大化对数似然：

$
  ell(theta) = log p(x|theta)
$

从某个参数 $theta^((i))$ 和隐变量分布 $q^((i))(z)$ 开始.

== E 步

// maximize the ELBO with respect to $q(z^((i)))$:

最大化 $ell(theta^((i)))$ 的 ELBO，即取等条件

$
  q^((i+1))(z) = p_(theta^((i))) (z|x)
$

== M 步

更新参数 $theta^((i))$ 以最大化 ELBO，// 等价于进行一次 MLE.

$
  theta^((i+1)) = arg max_theta cal(F)(q^((i+1)), theta)
$

== 收敛

$
  ell(theta^((i+1))) >=_("ELBO") cal(F)(q^((i+1)), theta^((i+1))) >=_(arg max) cal(F)(q^((i+1)), theta^((i)))
  =_("满足取等条件") ell(theta^((i)))
$

因此对数似然单调不减.

== 例子：高斯混合模型 GMM

假设观测数据 $x$ 来自 $K$ 个高斯分布的混合：

$
  p_theta (x) = sum_(k=1)^K pi_k NN(x|mu_k, Sigma_k)
$

其中 $theta = {pi_k, mu_k, Sigma_k}_(k=1)^K$ 是模型参数，$pi_k$ 是混合系数，满足 $sum_(k=1)^K pi_k = 1$.
数据样本为 ${x^(i)}_(i=1)^N$，引入隐变量 $z$ 表示样本属于哪个高斯分布, $z^(i) = k$ 表示样本 $x^(i)$ 来自第 $k$ 个高斯分布.

=== E 步

计算后验概率（责任度）：

$
  gamma_(z^(i)=k) &<- p_(theta) (z^(i)=k|x^(i)) \
  &= (pi_k NN(x^(i)|mu_k, Sigma_k)) / (sum_(j=1)^K pi_j NN(x^(i)|mu_j, Sigma_j))
$

=== M 步

更新参数：

$
  theta = arg max _theta cal(F)(gamma, theta)
$

即在已知责任度 $gamma_(z^(i)=k)$ 下，最大化似然的参数. 定义 $N_k & = sum_(i=1)^N gamma_(z^(i)=k)$,

$
  pi_k &<- N_k / N \
  mu_k &<- (1 / N_k) sum_(i=1)^N gamma_(z^(i)=k) x^(i) \
  Sigma_k &<- (1 / N_k) sum_(i=1)^N gamma_(z^(i)=k) (x^(i) - mu_k)(x^(i) - mu_k)^top \
$

= VAE

普通的自编码器（Autoencoder）就是编码–解码：

+ 编码器把输入数据 $x$ 映射到一个隐空间向量 $z$；
+ 解码器把 $z$ 还原回数据空间，得到重构 $hat(x)$.

但是，普通自编码器学习到的 $z$ 没有概率解释，不能直接采样用于生成.

VAE 的关键思想是：

给隐变量 $z$ 加一个概率分布的解释，并用变分推断来学习这个分布.
+ 编码器不再输出一个确定向量，而是输出一个 分布参数（均值 $mu(x)$、方差 $sigma^2(x)$）
+ 从这个分布中采样 $z ~ q_phi(z|x)$，再送给解码器生成 $hat(x)$.
+ 这样隐空间就被正则化成一个连续、平滑的概率空间，可以用来插值、采样、生成新样本.

== 先验分布 $p(z)$

隐变量 $z$ 的先验分布 $p(z)$ 通常取标准正态分布 $NN(0, I)$.

== 解码器（生成分布） $p_theta (x|z)$

网络参数 $theta$ 输入隐变量 $z$，输出数据 $x$ 的分布参数，即高斯分布的均值 $mu(z)$ 和方差 $Sigma(z)$.

== 编码器（近似后验分布）$q_phi (z|x)$

使用 $q_phi (z|x)$ 拟合“真实”后验分布 $p_theta (z|x)$.同样使用神经网络参数 $phi$，输入数据 $x$，输出隐变量 $z$ 的分布参数.

=== “近似”后验 和 “真实”后验

“真实”的后验分布由贝叶斯公式得出

$
  p_theta (x|z) = (p_theta (x|z) p(z)) / (p_theta (x)) \
  p_theta (x) = integral p_theta (x|z) p(z) dif z
$

因此真实的后验不可解，使用近似后验 $q_phi (z|x)$ 来代替.

== ELBO和损失函数

使用 ELBO 替代对数似然：

$
  log p_theta (x)>= EE_(q_phi (z|x)) [log p_theta (x|z)] - "KL"(q_phi (z|x) || p(z)) \
$

其中

- $EE_(q_phi (z|x)) [log p_theta (x|z)]$ 是重构误差，衡量解码器重构 $hat(x)$ 与输入 $x$ 的差异；
- $"KL"(q_phi (z|x) || p(z))$ 是正则化项，使近似后验分布 $q_phi (z|x)$ 尽量接近先验分布 $p(z)$（标准正态分布）.

== 训练和重参数化

目前我们有

+ 编码器 $q_phi (z|x)$，输入 $x$，输出隐变量 $z$ 的高斯分布参数 $mu(x), sigma^2(x)$；
+ 解码器 $p_theta (x|z)$，输入隐变量 $z$，输出重构 $hat(x)$ 的分布；
+ 先验分布 $p(z)$，通常取标准正态分布 $NN(0, I)$.
+ 损失函数 ELBO
  $
    cal(L) = -EE_(q_phi (z|x)) [log p_theta (x|z)] + "KL"(q_phi (z|x) || p(z))
  $

训练时，期望项（重构误差）没有解析解，使用 $z$ 的采样来近似. 为了保证采样 $z ~ NN(mu(x), sigma^2(x))$ 可导，使用重参数化技巧：

$
  z = mu(x) + sigma(x) dot.circle epsilon, \ epsilon ~ NN(0, I) "是独立噪声".
$

== 展开 KL 散度

记 $q_"agg" (x) = EE_(x~P(x)) q(z|x)$,

$
  "ELBO" &= EE_(q_phi (z|x)) [log p_theta (x|z)] - "KL"(q_"agg" (z) || p(z)) \
  &= underbrace(EE_(q_phi (z|x)) [log p_theta (x|z)], "重构项")
  - underbrace(H(q_"agg" (z), p(z)), "交叉熵")
  + underbrace(H(p(z)), "熵")
$

+ 重构项：鼓励隐变量分布远离先验分布 $p(z)$，提高重构质量
+ 交叉熵：把隐变量分布拉向先验分布 $p(z)$ 中心，会同时减小均值和方差
+ 熵：鼓励隐变量分布增大方差，分布更扁平

== VAE 的问题

=== Prior Hole

VAE 的隐空间分布 $q_"agg" (z)$ 往往只占据先验分布 $p(z)$ 的一小部分，导致从先验分布采样的 $z$ 很可能落在“空洞”区域，解码器无法生成有效样本.

=== Posterior Collapse

如果解码器足够强大，例如 autoregressive 模型，可以不依赖 $z$, 或者仅从 $z$ 的一部分维度中重构输入 $x$. 即 $z$ 的某些维度对重构没有贡献，导致这些维度的近似后验分布 $q_phi (z|x)$ 退化为先验分布 $p(z)$，从而无法学习到有效的隐表示.

$
  exists i space s.t. space forall x, q_phi (z_i|x) = p(z_i)
$

== Vector Quantized VAE(VQ VAE)

让隐变量 $z$ 取离散值，而不是连续值. 定义一个有限的码本（codebook），编码器输出一个向量 $e_i$，然后将 $e_i$ 映射到码本中距离最近的离散向量 $e_k$.

解决了 Prior Hole 和 Posterior Collapse 问题，但训练时需要使用特殊的技巧（如直通估计器）来处理离散变量的不可导问题.