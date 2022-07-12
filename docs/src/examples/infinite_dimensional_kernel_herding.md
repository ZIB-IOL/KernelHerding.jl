```@meta
EditURL = "<unknown>/examples/infinite_dimensional_kernel_herding.jl"
```

````@example infinite_dimensional_kernel_herding
using FrankWolfe
using KernelHerding
using Plots

include(joinpath(dirname(pathof(FrankWolfe)), "../examples/plot_utils.jl"))
````

# Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting

In this example, we illustrate how the Frank-Wolfe algorithm can be applied to infinite-dimensional kernel herding problems.
We first introduce the general kernel herding setting before discussing kernel herding for a specific infinite-dimensional kernel.

## Kernel herding

Kernel herding is known to be equivalent to solving a quadratic optimization problem in a
Reproducing Kernel Hilbert Space (RKHS) with the
Frank-Wolfe algorithm. We first explain kernel herding in general following [the paper](https://arxiv.org/pdf/2205.12838.pdf).
Let $\mathcal{Y} \subseteq \mathbb{R}$ be an observation space, $\mathcal{H}$ a RKHS with
inner product $\langle \cdot, \cdot \rangle_\mathcal{H}$, and
$\Phi \colon \mathcal{Y} \to \mathcal{H}$ a feature map such that any element $x\in \mathcal{H}$ has an associated real
function defined via

```math
x(y) = \langle x, \Phi(y)\rangle_\mathcal{H}
```
for $y\in \mathcal{Y}$. The feasible region in kernel herding is usually the marginal polytope $\mathcal{C}\subseteq \mathcal{H}$, which
is defined via
```math
\mathcal{C} : = \text{conv}\left(\lbrace \Phi(y) \mid y \in \mathcal{Y} \rbrace\right) \subseteq \mathcal{H}.
```
We then consider a probability distribution $\rho(y)$ over $\mathcal{Y}$ with associated mean element
```math
\mu : = \mathbb{E}_{\rho(y)} \Phi(y)(z) = \int_{\mathcal{Y}} k(z, y) \rho(y) dy \in \mathcal{C},
```
where $\mu in \mathcal{C}$ holds because the support of $p(y)$ is in $\mathcal{Y}$. [Bach et al.](https://icml.cc/2012/papers/683.pdf)
proved that kernel herding is equivalent to solving the minimization problem
```math
\min_{x\in \mathcal{C}} f(x), \qquad \qquad (OPT-KH)
```
where $f(x) := \frac{1}{2} \left\|x - \mu \right\|_\mathcal{H}^2$, with the Frank-Wolfe algorithm and open loop step-size rule
$\eta_t = \frac{1}{t + 1}$.
The well-definedness of kernel herding is guaranteed if there exists a constant $R > 0$ such that
$\|\Phi(y)\|_\mathcal{H} = R$ for all $y\in \mathcal{Y}$.
Moreover, all extreme points of $\mathcal{C}$ are of the form $\Phi(y)$ for $y\in \mathcal{Y}$. Thus, iterates constructed with the
Frank-Wolfe algorithm
are convex combinations of the form $x_t = \sum_{i=1}^t w_i \Phi(y_i)$, where $w =(w_1, \ldots, w_t)^\intercal \in \mathbb{R}^t$
is a weight
vector such that $w_i \geq 0$ for all $i \in \{1, \ldots, t}$ and $\sum_{i=1}^t w_i = 1$.

The iterate
$x_t$ is the mean element of the associated empirical distribution $\tilde{p}_t(y)$ over $\mathcal{Y}$, that is,
```math
\tilde{\mu}_t(z) = \mathbb{E}_{\tilde{\rho}_t(y)}\Phi(y)(z) = \sum_{i=1}^tw_i \Phi(y_i)(z) = x_t(z).
```
Then,
```math
\sup_{x\in \mathcal{H}, \|x\|_\mathcal{H} = 1} \lvert \mathbb{E}_{\rho (y)} x(y) - \mathbb{E}_{\tilde{\rho}_t(y)} x(y) \rvert = \|\mu - \tilde{\mu}_t\|_\mathcal{H}.
```
Thus, with kernel herding, by finding a good bound on $\|\mu - \tilde{\mu}_t\|_\mathcal{H}$, we can bound the error when computing the
expectation of $x\in \mathcal{H}$ with $\|x\|_\mathcal{H} = 1$.
## Infinite-dimensional kernel herding
Now that we have introduced the general kernel herding setting, we focus on a specific kernel studied in [Bach et al.](https://icml.cc/2012/papers/683.pdf)
and [Wirth et al.](https://arxiv.org/pdf/2205.12838.pdf).
Let $\mathcal{Y} = [0, 1]$ and
```math
\mathcal{H}:= \left\lbrace x \colon [0, 1] \to \mathbb{R} \mid x(y) = \sum_{j = 1}^\infty (a_j \cos(2\pi j y) + b_j \sin(2 \pi j y)), x'(y) \in L^2([0,1]), \text{ and } a_j,b_j\in\mathbb{R}\right\rbrace.
```
From now on, we will write $[0, 1]$ instead of $\mathcal{Y}$ to keep notation light. For $w, x \in \mathcal{H}$,
```math
\langle w, x \rangle_\mathcal{H} := \int_{[0,1]} w'(y)x'(y)dy
```
is an inner product. Thus, $(\mathcal{H}, \langle \cdot, \cdot \rangle_{\mathcal{H}})$ is a Hilbert space.
[Wirth et al.](https://arxiv.org/pdf/2205.12838.pdf) showed that $\mathcal{H}$ is a Reproducing Kernel Hilbert Space (RKHS) with
associated kernel
```math
k(y, z) = \frac{1}{2} B_2(| y - z |),
```
where $y,z\in [0, 1]$ and $B_2(y) = y^2 - y + \frac{1}{6}$ is the Bernoulli polynomial.

### Set-up
Below, we compare different Frank-Wolfe algorithm versions for kernel herding in the Hilbert space $\mathcal{H}$.
We always compare the Frank-Wolfe algorithm with open loop step-size rule $\eta_t = \frac{1}{t+1}$ (FW-OL),
Frank-Wolfe algorithm with short-step (FW-SS), and the Blended Pairwise Frank-Wolfe algorithm with short-step (BPFW-SS).
We do not use line search because it is equivalent to the short-step for the squared loss used in kernel herding.

The LMO in the here-presented kernel herding problem is implemented using exhaustive search over $\mathcal{Y} = [0, 1]$.

````@example infinite_dimensional_kernel_herding
max_iterations = 10000
max_iterations_lmo = 2 * max_iterations
lmo = MarginalPolytopeWahba(max_iterations_lmo)
````

### Uniform Distribution
First, we consider the uniform distribution $\rho = 1$, which results in the mean element being zero, that is, $\mu = 0$.

````@example infinite_dimensional_kernel_herding
rho = ([0.1, 0.2, 0.3], [0., 0., 3., 0.1])
normalized_rho = construct_rho(rho)
mu = mu_from_rho(normalized_rho)
iterate = KernelHerdingIterate([1.0], [0.0])
gradient = KernelHerdingGradient(iterate, mu)
f, grad = create_loss_function_gradient(mu)
````

f(iterate)
grad(gradient, iterate)

v = FrankWolfe.compute_extreme_point(lmo, gradient)
gamma = 1/2
println("f iterate: ", f(iterate))
updated = iterate + gamma * (v - iterate)
println("updated: ", updated)
println("f next: ", f(iterate + gamma * (v - iterate)))
println("dual gap: ", dot(gradient, iterate-v))
print(v)

````@example infinite_dimensional_kernel_herding
iterations = 1000

BPFW_SS = FrankWolfe.blended_pairwise_conditional_gradient(f, grad, lmo, iterate, line_search=FrankWolfe.Shortstep(1), verbose=true, gradient=gradient, memory_mode=FrankWolfe.OutplaceEmphasis(), max_iteration=iterations, trajectory=true)
FW_SS = FrankWolfe.frank_wolfe(f, grad, lmo, iterate, line_search=FrankWolfe.Shortstep(1), verbose=true, gradient=gradient, memory_mode=FrankWolfe.OutplaceEmphasis(), max_iteration=iterations, trajectory=true)
FW_OL = FrankWolfe.frank_wolfe(f, grad, lmo, iterate, line_search=FrankWolfe.Agnostic(), verbose=true, gradient=gradient, memory_mode=FrankWolfe.OutplaceEmphasis(), max_iteration=iterations, trajectory=true)
data = [BPFW_SS[end - 1], FW_SS[end], FW_OL[end]]
labels = ["BPFW-SS", "FW-SS", "FW-Ol"]
plot_trajectories(data, labels, xscalelog=true)
````

active_set = franky[end]
print(active_set)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

