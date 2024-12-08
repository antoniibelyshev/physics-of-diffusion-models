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