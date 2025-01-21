# Parameters

$\beta_t$ - parameters. In the current model: $\beta_1$ = `1e-4`, $\beta_N$ = `2e-2`, $\beta_t$ are linearly spaced for $1 \le t \le N$. $N$ is the number of steps in diffusion (currently $N = 1000$).
$$\alpha_t = 1 - \beta_t,\
\bar\alpha_t = \alpha_t\bar\alpha_{t - 1},\
\bar\alpha_0 = 1$$
$$T = \frac{1 - \bar\alpha_t}{\bar\alpha_t}$$

## Analytical approximation

If we suppose $N \to\infty$
$$\log\beta_{\alpha N} =
\sum\limits_{k = 1}^{\alpha N} \log\left( 1 - \beta_{min} - \frac{k\beta_{max}}{N} \right) =
-\alpha N \beta_\min - \alpha^2N\beta_\max$$
$$T = \exp(\beta_N t^2 / (2N)) - 1$$
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
