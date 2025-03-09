# Parameters

$\beta_t$ - parameters. In the current model: $\beta_1$ = `1e-4`, $\beta_N$ = `2e-2`, $\beta_t$ are linearly spaced for $1 \le t \le N$. $N$ is the number of steps in diffusion (currently $N = 1000$).
$$\alpha_t = 1 - \beta_t,\
\bar\alpha_t = \alpha_t\bar\alpha_{t - 1},\
\bar\alpha_0 = 1$$
$$T = \frac{1 - \bar\alpha_t}{\bar\alpha_t}$$

## Analytical approximation

If we suppose $N \to\infty$
$$\log\frac{\bar\alpha_{\tau N}}{1 - \beta_0} =
\sum\limits_{k = 1}^{\tau N} \log\left( 1 - \frac{k}{N}\frac{\beta_N - \beta_0}{1 - \beta_0} \right) =
-\frac{\tau^2 N}{2}\frac{\beta_N - \beta_0}{1 - \beta_0}$$
$$T = \frac{\exp(\frac{\tau^2N}{2}\frac{\beta_N - \beta_0}{1 - \beta_0})}{1 - \beta_0} - 1$$
$$N = 1000, \beta_N = 0.02$$

# Variance preserving scheme

$$z = \sqrt{\bar\alpha_t}y + \sqrt{1 - \bar\alpha}\epsilon$$

$$p(y|z, t) =
\phi(y)\frac{p(z|y, t)}{p(z, t)} =
\phi(y)\frac{N(z|\sqrt{\bar\alpha_t}y, (1 - \bar\alpha_t)I)}{p(z, t)} =
\phi(y)\frac{(2\pi (1 - \bar\alpha_t))^{-d/2}}{p(z, t)}
\exp\left(-\frac{||z - \sqrt{\bar\alpha_t} y||^2}{2(1 - \bar\alpha_t)}\right) =
\phi(y)\frac{\exp\left(-\beta H_z(y|z, t)\right)}{Z_z(z, t)}$$

$$T = \frac{1 - \bar\alpha_t}{\bar\alpha_t},\
\bar\alpha_t = \frac{1}{1 + T},\
1 - \bar\alpha_t = \frac{T}{1 + T}$$

$$H_z(y|z, t) =
\frac{1}{2}\lVert \sqrt{1 + T} z - y\rVert^2$$

$$Z_z(z, t) =
\int \phi(y)\exp\left(-\beta H_z(y|z, t)\right)dy =
(2\pi(1 - \bar\alpha_t))^{d/2}p(z, t)$$

# Linear noise scheme

$$x = y + \sqrt{T}\epsilon =
\frac{z}{\sqrt{\bar\alpha_t}}$$

$$p(y|x, t) =
\phi(y)\frac{p(x|y, t)}{p(z, t)} =
\phi(y)\frac{N(x|y, TI)}{p(x, t)} =
\phi(y)\frac{(2\pi T)^{-d/2}}{p(x, t)}
\exp\left(-\frac{||x - y||^2}{2T}\right) =
\phi(y)\frac{\exp\left(-\beta H(y|x, t)\right)}{Z(x, t)}$$

$$H_x(y|x, t) =
\frac{1}{2}\lVert x - y\rVert^2$$

$$Z_x(x, t) =
\int \phi(y)\exp\left(-\beta H_x(y|x, t)\right)dy =
(2\pi T)^{d/2}p(x, t) =
Z_z(\sqrt{\bar\alpha}, t)$$

# Free energy

$$F_x(x, t) = -T\log Z_x(x, t)$$
$$F_z(z, t) = -T\log Z_z(z, t)$$
$$F = - T\log Z$$

$$S = -\frac{dF}{dT},\ C = -T\frac{d^2F}{dT^2} = \frac{d}{d \log T} S$$

# Derivations

$$U(x, t) =
\mathbb{E}_{y \sim p(y|x, t)} H(y|x, t) =
\int \phi(y) \frac{\exp(-\beta H(y|x, t))}{Z(x, t)}H(y|x, t)dy$$
For $\phi(y)$ equal to sum of $N$ delta functions:
$$U(x, t) =
\frac{1}{N}\sum\limits_{i = 1}^N \frac{\exp(-\beta H(y_i|x, t))}{Z(x, t)}H(y_i|x, t)d y$$
Auxiliary computation:
$$\frac{\partial \log Z(x, t)}{\partial T} =
-\frac{1}{{Z(x, t)}}\frac{\partial}{\partial T} \int \phi(y)\exp(-\beta H(y|x, t))dy =
\frac{1}{T^2{Z(x, t)}}\int \phi(y)\exp(-\beta H(y|x, t))H(y|x, t)dy =
\frac{1}{T^2}U(x, t)$$

Entropy:
$$-\frac{\partial F(x, t)}{\partial T} = \frac{\partial}{\partial T}(T\log Z(x, t)) =
\log Z(x, t) + \frac{1}{T}U(x, t) =
-\mathbb{E}_{y \sim p(y|x, t)} \log\left( \frac{\exp(-\beta H(y|x, t))}{Z(x, t)} \right) = S + \mathbb{E}_{y \sim p(y|x, t)} \phi(y)$$

Heat capacity:
$$-T\frac{\partial^2 T}{\partial T^2} F(x, t) =
-T\frac{\partial}{\partial T}\frac{\partial F(x, t)}{\partial T} = -T \left( -\frac{1}{T^2}U(x, t) + \frac{1}{T^2}U(x, t) - \frac{1}{T}\frac{\partial U(x, t)}{\partial T} \right) =
\frac{\partial U(x, t)}{\partial T}$$

$$C = \frac{\partial U(x, t)}{\partial T} = \frac{\partial}{\partial T}\int\phi(y) \frac{\exp(-\beta H(y|x, t))}{Z(x, t)}H(y|x, t)dy = \frac{1}{T^2}\left(\mathbb{E}H(y,|x, t)^2 - \left(\mathbb{E}H(y|x, t)\right)^2\right) = \frac{1}{T^2}\mathbb{D}H(x, t)$$


# Included prior distribution

### Definitions
$$p(y|x, T) = \frac{\exp(-H(x, y, T) / T)}{Z(x, T)}$$
$$H(x, y, T) = \frac{||x - y||^2}{2} - T\log\phi(y),\ Z(x, T) = \int\exp(-H(x, y, T) / T)d y$$
$$U(x, T) = \mathbb{E}_{y|x, T} H(x, y, T),\ C(x, T) = \frac{\partial U}{\partial T}$$

### Derivations
$$S(x, T) = -\int \frac{\exp(-H(x, y, T) / T)}{Z(x, T)}\left( -H(x, y, T) / T - \log Z(x, T) \right)dy = \log Z(x, T) + U(x, T) / T$$
$$T\frac{\partial S}{\partial T} = T\frac{\partial \log Z(x, T)}{\partial T} - \frac{1}{T}U(x, T) + \frac{\partial U}{\partial T}$$
$$\frac{\partial \log Z(x, T)}{\partial T} = \frac{1}{T^2}U(x, T) + \frac{1}{T}\mathbb{E} \log\phi(y)$$
$$T\frac{\partial S}{\partial T} = \frac{1}{T}\mathbb{E} \log\phi(y) + C(x, T)$$
$$C(x, T) = \frac{\partial}{\partial T}\int \frac{\exp(-H(x, y, T) / T)}{Z(x, T)} H(x, y, T)dy =$$
$$\frac{1}{T^2}\mathrm{Var} H(x, y, T) - \frac{1}{T}U(x, T)\mathbb{E}\log\phi(y) - \frac{1}{T}\mathbb{E}\log\phi(y) + \frac{1}{T}\mathbb{E}\left(\log\phi(y)H(x, y, T)\right)$$
$$T\frac{\partial S}{\partial T} = \frac{1}{T^2}\mathrm{Var} H(x, y, T) - \frac{1}{T}U(x, T)\mathbb{E}\log\phi(y) + \frac{1}{T}\mathbb{E}\left(\log\phi(y)H(x, y, T)\right) =$$
$$\frac{1}{T^2}\mathrm{Var}H(x, y, T) + \frac{1}{T}\mathrm{Cov}\left(\frac{||x - y||}{2}, \log\phi(y)\right) + \mathrm{Var} \log\phi(y)$$

# $T \to \infty$

$$p(y|x, T) \approx \phi(y), p(x, T) \approx N(x|0, T)$$
$$U(x, T) = \int p(y|x, T) \frac{||x - y||^2}{2}dy \approx \frac{x^2}{2}$$
$$\mathbb{E}_{x \sim p(x, T)} U(x, T) = \mathbb{E}_{x \sim p(x, T)} \frac{x^2}{2} \approx \frac{T}{2}$$
$$\frac{d}{dT} \mathbb{E}_{x \sim p(x, T)} U(x, T) \approx \frac{1}{2}$$

# Sampling

$$T(t) = \frac{1 - \bar\alpha_t}{\bar\alpha_t}$$
$$z_t = z_0 + \sqrt{T(t)}\epsilon_t$$
$$s_z(z, t) = \nabla_z\log p_z(z, t) = -\frac{\epsilon_t}{\sqrt{T(t)}} = \frac{1}{T(t)}z_0 - \frac{1}{T(t)}z_t$$
$$z_t = \sqrt{1 + T(t)}x_t = \frac{x_t}{\sqrt{\bar\alpha_t}}$$

### DDPM

$$x_{t - 1} = \frac{\sqrt{\bar\alpha_{t - 1}}\beta_t}{1 - \bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t - 1})}{1 - \bar\alpha_t}x_t + \sqrt{\frac{1 - \bar\alpha_{t - 1}}{1 - \bar\alpha_t}\beta_t}\xi_t$$

$$z_{t - 1} = \frac{1}{\sqrt{\bar\alpha_{t - 1}}}\frac{\sqrt{\bar\alpha_{t - 1}}\beta_t}{1 - \bar\alpha_t}z_0 + \frac{\sqrt{\bar\alpha_t}}{\sqrt{\bar\alpha_{t - 1}}}\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t - 1})}{1 - \bar\alpha_t}z_t + \frac{1}{\sqrt{\bar\alpha_{t - 1}}}\sqrt{\frac{1 - \bar\alpha_{t - 1}}{1 - \bar\alpha_t}\beta_t}\xi_t =$$
$$\frac{\beta_t}{1 - \bar\alpha_t}z_0 + \frac{\alpha_t(1 - \bar\alpha_{t - 1})}{1 - \bar\alpha_t}z_t + \sqrt{\frac{\beta_t}{\bar\alpha_t}}\sqrt{\frac{1 - \bar\alpha_t - \beta_t}{1 - \bar\alpha_t}}\xi_t = z_t + \frac{\beta_t}{\bar\alpha_t}s_z(z, t) + \sqrt{\frac{\beta_t}{\bar\alpha_t}}\sqrt{1 - \frac{\beta_t}{1 - \bar\alpha_t}}\xi_t$$

### DDIM

$$x_{t - 1} = \sqrt{\bar\alpha_{t - 1}}x_0 + \sqrt{1 - \bar\alpha_{t - 1}}\epsilon_\theta = \sqrt{\bar\alpha_{t - 1}}x_0 + \sqrt{1 - \bar\alpha_{t - 1}}\frac{x_t - \sqrt{\bar\alpha_t}x_0}{\sqrt{1 - \bar\alpha_t}}$$
$$z_{t - 1} = z_0 + \sqrt{\frac{T(t - 1)}{T(t)}}(z_t - z_0) = z_t + \left( \sqrt{\frac{T(t - 1)}{T(t)}} - 1 \right)T(t)s_z(z, t)$$
$$\sqrt{\frac{T(t - 1)}{T(t)}} - 1 = \sqrt{\alpha_t\frac{1 - \bar\alpha_{t - 1}}{1 - \bar\alpha_t}} - 1 = \sqrt{1 - \frac{\beta_t}{1 - \bar\alpha_t}} - 1$$

# Low temperature limit

$$S(x_t|T) = \log Z(x_t|T) + \frac{U(x_t|T)}{T}$$
$$S(T) = \mathbb{E}_{x_t \sim p(x_t|T)} S(x_t|T)$$

$$p(x_0 = x^{(k)}|x_t, T) =
\frac{p(x_t|x_0 = x^{(k)}, T)}{\sum\limits_{i = 1}^N p(x_t|x_0 = x^{(i)}, T)} =
\frac{\exp\left(-\frac{||x_t - x^{(k)}||^2}{2T}\right)}{\sum\limits_{i = 1}^N \exp\left(-\frac{||x_t - x^{(i)}||^2}{2T}\right)}$$

$$x_t = x^{(m)} + \sqrt{T}\xi$$
$$\log Z(x_t, T) =
\log\sum\limits_{i = 1}^N \exp\left(-\frac{||x_t - x^{(i)}||^2}{2T}\right)=
-\frac{||\xi||^2}{2} + \log\left( 1 + \sum\limits_{i \ne m} \exp\left( -\frac{||x^{(i)} - x^{(m)} + \xi\sqrt{T}||^2}{2T} + \frac{||\xi||^2}{2} \right) \right) =$$
$$-\frac{||\xi||^2}{2} + \log\left(1 + \sum\limits_{i \ne m} \exp\left(-\frac{||x^{(i)} - x^{(m)}||^2}{2T}\right)\exp\left(\frac{(x^{(m)} - x^{(i)})\cdot\xi}{\sqrt{T}}\right)\right)$$

$$\mathbb{E}_{\eta\sim\mathcal{N}(\mu, \sigma^2)}\log (1 + \exp(\eta)) \approx
\int\limits_0^\infty \mathcal{N}(\eta|\mu, \sigma^2)\eta\mathrm{d}\eta =
\sigma\int\limits_{-\mu / \sigma}^\infty \frac{1}{\sqrt{2\pi}} \exp(-\zeta^2 / 2)\zeta\mathrm{d}\zeta =$$
$$[\text{assuming } \mu < 0, \text{ denoting } z = \zeta^2 / 2] =
\frac{\sigma}{\sqrt{2\pi}}\int\limits_{\mu^2 / (2\sigma^2)}^\infty e^{-z}\mathrm{d}z =
\frac{\sigma}{\sqrt{2\pi}} \exp(-\mu^2 / (2\sigma^2))$$

$$s(X, T) =
\mathbb{E}_{\xi\sim\mathcal{N}(0, I)} \log\left( 1 + \sum\limits_{i = 1, i\ne m}^N \exp\left( -\frac{||x^{(i)} - x^{(m)}||^2}{2T} + \frac{(x^{(i)} - x^{(m)})\cdot\xi}{\sqrt{T}} \right) \right)$$

Assume that either the sum is very small, or it is dominated by the term with the lowest value of $||x^{(i)} - x^{(m)}||$. Then denote

$$j = \arg \min\limits_{i} ||x^{(i)} - x^{(m)}||,\
\delta = ||x^{(j)} - x^{(m)}||,\
\eta = -\frac{\delta^2}{2T} + \frac{(x^{(i)} - x^{(m)})\cdot\xi}{\sqrt{T}}$$
$$\eta \sim\mathcal{N}\left( -\frac{\delta^2}{2T}, \frac{\delta^2}{T} \right)$$

$$s(X, T) \approx \mathbb{E}_\eta \log(1 + \exp(\eta)) \approx
\frac{\delta}{\sqrt{2\pi T}} \exp(-\delta^2 / (8T))$$

When $N \to \infty$
$$P(\delta > r) \approx
(1 - V_dr^d\phi\left(x^{(m)}\right))^N \approx
\exp(-V_dr^d\phi\left(x^{(m)}\right)N)$$
$$p(\delta) = cd\delta^{d - 1}\exp(-c\delta^d),\
c = V_d\phi\left(x^{(m)}\right)N$$
Where $V_d$ is a volume of a unit ball in $\mathbb{R}^d$, $\phi$ is a population distribution.

$$\mathbb{E}_X s(X, T) \approx
\mathbb{E}_{x^{(m)}}\mathbb{E}_{\delta \sim p(\delta)} \frac{\delta}{\sqrt{2\pi}}\exp(-\delta^2 / (8T)) \approx \left[\delta \approx \sqrt{T} \ll \frac{1}{c^{(1/d)}}\right] \approx$$
$$\mathbb{E}_{x^{(m)}}\int\limits_0^\infty \frac{cd\delta^d}{\sqrt{2\pi}}\exp(-\delta^2 / (8T))\mathrm{d}\delta =
\left[ \Gamma(z) = 2c^z \int\limits_0^\infty t^{2z - 1}\exp(-ct^2)\mathrm{d}t \right] =
\mathbb{E}_{x^{(m)}}\frac{cd}{\sqrt{2\pi}}\frac{\Gamma((d + 1) / 2)}{2}(8T)^{(d + 1) / 2} =$$
$$[\text{substituting the expression for } c] =
\mathbb{E}_{x^{(m)}}\left(\phi\left(x^{(m)}\right)\right) \frac{\pi^{d / 2}d}{2\sqrt{2\pi}}N(8T)^{(d + 1) / 2}$$
