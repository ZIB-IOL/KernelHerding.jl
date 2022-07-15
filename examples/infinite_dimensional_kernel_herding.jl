using FrankWolfe
using KernelHerding
using Plots
using LinearAlgebra
using Random

include(joinpath(dirname(pathof(FrankWolfe)), "../examples/plot_utils.jl"))


# # Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting

# In this example, we illustrate how the Frank-Wolfe algorithm can be applied to infinite-dimensional kernel herding problems.
# First, we present a quick primer on kernel herding.

# ## Kernel herding

# Kernel herding is known to be equivalent to solving a quadratic optimization problem in a
# Reproducing Kernel Hilbert Space (RKHS) with the 
# Frank-Wolfe algorithm ([Bach et al.](https://icml.cc/2012/papers/683.pdf)). Here, we explain kernel herding following the presentation of [Wirth et al.](https://arxiv.org/pdf/2205.12838.pdf).

# Let $\mathcal{Y} \subseteq \mathbb{R}$ be an observation space, $\mathcal{H}$ a RKHS with 
# inner product $\langle \cdot, \cdot \rangle_\mathcal{H}$, and 
# $\Phi \colon \mathcal{Y} \to \mathcal{H}$ a feature map such that any element $x\in \mathcal{H}$ has an associated real
# function defined via
#
# ```math
# x(y) = \langle x, \Phi(y)\rangle_\mathcal{H}
# ```
# for $y\in \mathcal{Y}$. Here, the feasible region is the marginal polytope 
# ```math
# \mathcal{C} : = \text{conv}\left(\lbrace \Phi(y) \mid y \in \mathcal{Y} \rbrace\right) \subseteq \mathcal{H}.
# ```
# We consider a probability distribution $\rho(y)$ over $\mathcal{Y}$ with associated mean element 
# ```math
# \mu(z) : = \mathbb{E}_{\rho(y)} \Phi(y)(z) = \int_{\mathcal{Y}} k(z, y) \rho(y) dy \in \mathcal{C},
# ```
# where $\mu \in \mathcal{C}$ due to the support of $p(y)$ being in $\mathcal{Y}$. 
# Then, kernel herding is equivalent to solving the minimization problem
# ```math
# \min_{x\in \mathcal{C}} f(x), \qquad \qquad \text{(OPT-KH)}
# ```
# where $f(x) := \frac{1}{2} \left\|x - \mu \right\|_\mathcal{H}^2$, with the Frank-Wolfe algorithm and open loop step-size rule 
# $\eta_t = \frac{1}{t + 1}$. 
# The well-definedness of kernel herding is guaranteed if there exists a constant $R > 0$ such that
# $\|\Phi(y)\|_\mathcal{H} = R$ for all $y\in \mathcal{Y}$.
# In that case, all extreme points of $\mathcal{C}$ are of the form $\Phi(y)$ for $y\in \mathcal{Y}$. Thus, iterates constructed with the
# Frank-Wolfe algorithm
# are convex combinations of the form $x_t = \sum_{i=1}^t w_i \Phi(y_i)$, where $w =(w_1, \ldots, w_t)^\intercal \in \mathbb{R}^t$
# is a weight
# vector such that $w_i \geq 0$ for all $i \in \{1, \ldots, t\}$ and $\sum_{i=1}^t w_i = 1$. 
# Observe that the iterate 
# $x_t$ is the mean element of the associated empirical distribution $\tilde{\rho}_t(y)$ over $\mathcal{Y}$, that is,
# ```math
# \tilde{\mu}_t(z) = \mathbb{E}_{\tilde{\rho}_t(y)}\Phi(y)(z) = \sum_{i=1}^tw_i \Phi(y_i)(z) = x_t(z).
# ```
# Then,
# ```math
# \sup_{x\in \mathcal{H}, \|x\|_\mathcal{H} = 1} \lvert \mathbb{E}_{\rho (y)} x(y) - \mathbb{E}_{\tilde{\rho}_t(y)} x(y) \rvert = \|\mu - \tilde{\mu}_t\|_\mathcal{H}.
# ```
# Thus, with kernel herding, by finding a good bound on $\|\mu - \tilde{\mu}_t\|_\mathcal{H}$, we can bound the error when computing the 
# expectation of $x\in \mathcal{H}$ with $\|x\|_\mathcal{H} = 1$.
# ## Infinite-dimensional kernel herding
# Now that we have introduced the general kernel herding setting, we focus on a specific kernel studied in [Bach et al.](https://icml.cc/2012/papers/683.pdf)
# and [Wirth et al.](https://arxiv.org/pdf/2205.12838.pdf). 
# Let $\mathcal{Y} = [0, 1]$ and
# ```math
# \mathcal{H}:= \left\lbrace x \colon [0, 1] \to \mathbb{R} \mid x(y) = \sum_{j = 1}^\infty (a_j \cos(2\pi j y) + b_j \sin(2 \pi j y)), x'(y) \in L^2([0,1]), \text{ and } a_j,b_j\in\mathbb{R}\right\rbrace.
# ```
# From now on, we will write $[0, 1]$ instead of $\mathcal{Y}$ to keep notation light. For $w, x \in \mathcal{H}$, 
# ```math
# \langle w, x \rangle_\mathcal{H} := \int_{[0,1]} w'(y)x'(y)dy
# ```
# is an inner product. Thus, $(\mathcal{H}, \langle \cdot, \cdot \rangle_{\mathcal{H}})$ is a Hilbert space. 
# [Wirth et al.](https://arxiv.org/pdf/2205.12838.pdf) showed that $\mathcal{H}$ is a Reproducing Kernel Hilbert Space (RKHS) with
# associated kernel
# ```math
# k(y, z) = \frac{1}{2} B_2(| y - z |),
# ```
# where $y,z\in [0, 1]$ and $B_2(y) = y^2 - y + \frac{1}{6}$ is the Bernoulli polynomial.

# ### Set-up
# Below, we compare different Frank-Wolfe algorithm versions for kernel herding in the Hilbert space $\mathcal{H}$.
# We always compare the Frank-Wolfe algorithm with open loop step-size rule $\eta_t = \frac{1}{t+1}$ (FW-OL),
# Frank-Wolfe algorithm with short-step (FW-SS), and the Blended Pairwise Frank-Wolfe algorithm with short-step (BPFW-SS).
# We do not use line search because it is equivalent to the short-step for the squared loss used in kernel herding.

# The LMO in the here-presented kernel herding problem is implemented using exhaustive search over $\mathcal{Y} = [0, 1]$, which we perform
# for twice the number of iterations we run the Frank-Wolfe algorithms for. 

max_iterations = 1000
max_iterations_lmo = 2 * max_iterations
lmo = MarginalPolytopeWahba(max_iterations_lmo)

# ### Uniform distribution
# First, we consider the uniform distribution $\rho = 1$, which results in the mean element being zero, that is, $\mu = 0$.

mu = ZeroMeanElement()
iterate = KernelHerdingIterate([1.0], [0.0])
gradient = KernelHerdingGradient(iterate, mu)
f, grad = create_loss_function_gradient(mu)

FW_OL = FrankWolfe.frank_wolfe(f, grad, lmo, iterate, line_search=FrankWolfe.Agnostic(), verbose=true, gradient=gradient, memory_mode=FrankWolfe.OutplaceEmphasis(), max_iteration=max_iterations, trajectory=true)
FW_SS = FrankWolfe.frank_wolfe(f, grad, lmo, iterate, line_search=FrankWolfe.Shortstep(1), verbose=true, gradient=gradient, memory_mode=FrankWolfe.OutplaceEmphasis(), max_iteration=max_iterations, trajectory=true)
BPFW_SS = FrankWolfe.blended_pairwise_conditional_gradient(f, grad, lmo, iterate, line_search=FrankWolfe.Shortstep(1), verbose=true, gradient=gradient, memory_mode=FrankWolfe.OutplaceEmphasis(), max_iteration=max_iterations, trajectory=true)
data = [FW_OL[end], FW_SS[end], BPFW_SS[end - 1]]
labels = ["FW-OL", "FW-SS", "BPFW-SS"]
plot_trajectories(data, labels, xscalelog=true)

# Observe that FW-OL converges faster than FW-SS and BPFW-SS. [Wirth et al.](https://arxiv.org/pdf/2205.12838.pdf) proved the accelerated convergence
# rate of $\mathcal{O}(1/t^2)$ for FW-OL, but it remains an open problem to prove that FW-SS and BPFW-SS do not admit this accelerated rate.  

# ### Non-uniform distribution
# Second, we consider a non-uniform distribution 
# ```math
# \rho(y) \backsim \left(\sum_{i = 1}^n a_i \cos(2\pi i y) + b_i \sin (2\pi i y) \right)^2,
# ```
# where $n\in \mathbb{N}$, $a_i,b_i \in \mathbb{R}$ for all $i \in \{1, \ldots, n}$, and $a_i, b_i$ are chosen such that 
# ```math
# \int_{\mathcal{Y}} \rho(y) dy = 1.
# ```
# To obtain such a $\rho$, we start with an arbitrary tuple of vectors:

rho = (rand((1, 5)), rand((1, 8)))


# We then normalize the vectors to obtain a $\rho$ that is indeed a distribution.
normalized_rho = construct_rho(rho)

# We then run the experiments.
mu = mu_from_rho(normalized_rho)
iterate = KernelHerdingIterate([1.0], [0.0])
gradient = KernelHerdingGradient(iterate, mu)
f, grad = create_loss_function_gradient(mu)



FW_OL = FrankWolfe.frank_wolfe(f, grad, lmo, iterate, line_search=FrankWolfe.Agnostic(), verbose=true, gradient=gradient, memory_mode=FrankWolfe.OutplaceEmphasis(), max_iteration=max_iterations, trajectory=true)
FW_SS = FrankWolfe.frank_wolfe(f, grad, lmo, iterate, line_search=FrankWolfe.Shortstep(1), verbose=true, gradient=gradient, memory_mode=FrankWolfe.OutplaceEmphasis(), max_iteration=max_iterations, trajectory=true)
BPFW_SS = FrankWolfe.blended_pairwise_conditional_gradient(f, grad, lmo, iterate, line_search=FrankWolfe.Shortstep(1), verbose=true, gradient=gradient, memory_mode=FrankWolfe.OutplaceEmphasis(), max_iteration=max_iterations, trajectory=true)
data = [FW_OL[end], FW_SS[end], BPFW_SS[end - 1]]
labels = ["FW-OL", "FW-SS", "BPFW-SS"]
plot_trajectories(data, labels, xscalelog=true)

# Observe that FW-OL converges with a rate of $\mathcal{O}(1/t^2)$, which is faster than the convergence rate of
# $\mathcal{O}(1/t)$ admitted by FW-SS and BPFW-SS. Explaining this phenomenon of acceleration remains an open problem.  














