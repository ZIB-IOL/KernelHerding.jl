var documenterSearchIndex = {"docs":
[{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"EditURL = \"https://github.com/ZIB-IOL/KernelHerding.jl/blob/main/examples/infinite_dimensional_kernel_herding.jl\"","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"using FrankWolfe\nusing KernelHerding\nusing Plots\nusing LinearAlgebra\nusing Random\n\ninclude(joinpath(dirname(pathof(FrankWolfe)), \"../examples/plot_utils.jl\"))","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/#Kernel-herding:-The-Frank-Wolfe-algorithm-in-an-infinite-dimensional-setting","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"","category":"section"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"In this example, we illustrate how the Frank-Wolfe algorithm can be applied to infinite-dimensional kernel herding problems. We first introduce the general kernel herding setting before discussing the specifics of the example setting.","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/#Kernel-herding","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding","text":"","category":"section"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"Kernel herding is known to be equivalent to solving a quadratic optimization problem in a Reproducing Kernel Hilbert Space (RKHS) with the Frank-Wolfe algorithm, as proved, e.g., in Bach et al.. Here, we explain kernel herding following the presentation of Wirth et al..","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"Let mathcalY subseteq mathbbR be an observation space, mathcalH a RKHS with inner product langle cdot cdot rangle_mathcalH, and Phi colon mathcalY to mathcalH a feature map such that any element xin mathcalH has an associated real function defined via","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"x(y) = langle x Phi(y)rangle_mathcalH","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"for yin mathcalY. The feasible region in kernel herding is usually the marginal polytope mathcalCsubseteq mathcalH, which is defined via","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"mathcalC  = textconvleft(lbrace Phi(y) mid y in mathcalY rbraceright) subseteq mathcalH","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"We consider a probability distribution rho(y) over mathcalY with associated mean element","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"mu  = mathbbE_rho(y) Phi(y)(z) = int_mathcalY k(z y) rho(y) dy in mathcalC","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"where mu in mathcalC is guaranteed because the support of p(y) is in mathcalY. Then, kernel herding is equivalent to solving the minimization problem","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"min_xin mathcalC f(x) qquad qquad (OPT--KH)","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"where f(x) = frac12 leftx - mu right_mathcalH^2, with the Frank-Wolfe algorithm and open loop step-size rule eta_t = frac1t + 1. The well-definedness of kernel herding is guaranteed if there exists a constant R  0 such that Phi(y)_mathcalH = R for all yin mathcalY. Moreover, all extreme points of mathcalC are of the form Phi(y) for yin mathcalY. Thus, iterates constructed with the Frank-Wolfe algorithm are convex combinations of the form x_t = sum_i=1^t w_i Phi(y_i), where w =(w_1 ldots w_t)^intercal in mathbbR^t is a weight vector such that w_i geq 0 for all i in 1 ldots t and sum_i=1^t w_i = 1. The iterate x_t is the mean element of the associated empirical distribution tildep_t(y) over mathcalY, that is,","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"tildemu_t(z) = mathbbE_tilderho_t(y)Phi(y)(z) = sum_i=1^tw_i Phi(y_i)(z) = x_t(z)","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"Then,","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"sup_xin mathcalH x_mathcalH = 1 lvert mathbbE_rho (y) x(y) - mathbbE_tilderho_t(y) x(y) rvert = mu - tildemu_t_mathcalH","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"Thus, with kernel herding, by finding a good bound on mu - tildemu_t_mathcalH, we can bound the error when computing the expectation of xin mathcalH with x_mathcalH = 1.","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/#Infinite-dimensional-kernel-herding","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Infinite-dimensional kernel herding","text":"","category":"section"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"Now that we have introduced the general kernel herding setting, we focus on a specific kernel studied in Bach et al. and Wirth et al.. Let mathcalY = 0 1 and","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"mathcalH= leftlbrace x colon 0 1 to mathbbR mid x(y) = sum_j = 1^infty (a_j cos(2pi j y) + b_j sin(2 pi j y)) x(y) in L^2(01) text and  a_jb_jinmathbbRrightrbrace","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"From now on, we will write 0 1 instead of mathcalY to keep notation light. For w x in mathcalH,","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"langle w x rangle_mathcalH = int_01 w(y)x(y)dy","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"is an inner product. Thus, (mathcalH langle cdot cdot rangle_mathcalH) is a Hilbert space. Wirth et al. showed that mathcalH is a Reproducing Kernel Hilbert Space (RKHS) with associated kernel","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"k(y z) = frac12 B_2( y - z )","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"where yzin 0 1 and B_2(y) = y^2 - y + frac16 is the Bernoulli polynomial.","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/#Set-up","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Set-up","text":"","category":"section"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"Below, we compare different Frank-Wolfe algorithm versions for kernel herding in the Hilbert space mathcalH. We always compare the Frank-Wolfe algorithm with open loop step-size rule eta_t = frac1t+1 (FW-OL), Frank-Wolfe algorithm with short-step (FW-SS), and the Blended Pairwise Frank-Wolfe algorithm with short-step (BPFW-SS). We do not use line search because it is equivalent to the short-step for the squared loss used in kernel herding.","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"The LMO in the here-presented kernel herding problem is implemented using exhaustive search over mathcalY = 0 1, which we perform for twice the number of iterations we run the Frank-Wolfe algorithms for.","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"max_iterations = 1000\nmax_iterations_lmo = 2 * max_iterations\nlmo = MarginalPolytopeWahba(max_iterations_lmo)","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/#Uniform-distribution","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Uniform distribution","text":"","category":"section"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"First, we consider the uniform distribution rho = 1, which results in the mean element being zero, that is, mu = 0.","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"mu = ZeroMeanElement()\niterate = KernelHerdingIterate([1.0], [0.0])\ngradient = KernelHerdingGradient(iterate, mu)\nf, grad = create_loss_function_gradient(mu)\n\nFW_OL = FrankWolfe.frank_wolfe(f, grad, lmo, iterate, line_search=FrankWolfe.Agnostic(), verbose=true, gradient=gradient, memory_mode=FrankWolfe.OutplaceEmphasis(), max_iteration=max_iterations, trajectory=true)\nFW_SS = FrankWolfe.frank_wolfe(f, grad, lmo, iterate, line_search=FrankWolfe.Shortstep(1), verbose=true, gradient=gradient, memory_mode=FrankWolfe.OutplaceEmphasis(), max_iteration=max_iterations, trajectory=true)\nBPFW_SS = FrankWolfe.blended_pairwise_conditional_gradient(f, grad, lmo, iterate, line_search=FrankWolfe.Shortstep(1), verbose=true, gradient=gradient, memory_mode=FrankWolfe.OutplaceEmphasis(), max_iteration=max_iterations, trajectory=true)\ndata = [FW_OL[end], FW_SS[end], BPFW_SS[end - 1]]\nlabels = [\"FW-OL\", \"FW-SS\", \"BPFW-SS\"]\nplot_trajectories(data, labels, xscalelog=true)","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"Observe that FW-OL converges faster than FW-SS and BPFW-SS. Wirth et al. proved the accelerated convergence rate of mathcalO(1t^2) for FW-OL, but it remains an open problem to prove that FW-SS and BPFW-SS do not admit this accelerated rate.","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/#Non-uniform-distribution","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Non-uniform distribution","text":"","category":"section"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"Second, we consider a non-uniform distribution","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"rho(y) backsim left(sum_i = 1^n a_i cos(2pi i y) + b_i sin (2pi i y) right)^2","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"where nin mathbbN, a_ib_i in mathbbR for all i in 1 ldots n, and a_i b_i are chosen such that","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"int_mathcalY rho(y) dy = 1","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"To obtain such a rho, we start with an arbitrary tuple of vectors:","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"rho = (rand((1, 5)), rand((1, 8)))","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"We then normalize the vectors to obtain a rho that is indeed a distribution.","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"normalized_rho = construct_rho(rho)","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"We then run the experiments.","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"mu = mu_from_rho(normalized_rho)\niterate = KernelHerdingIterate([1.0], [0.0])\ngradient = KernelHerdingGradient(iterate, mu)\nf, grad = create_loss_function_gradient(mu)\n\n\n\nFW_OL = FrankWolfe.frank_wolfe(f, grad, lmo, iterate, line_search=FrankWolfe.Agnostic(), verbose=true, gradient=gradient, memory_mode=FrankWolfe.OutplaceEmphasis(), max_iteration=max_iterations, trajectory=true)\nFW_SS = FrankWolfe.frank_wolfe(f, grad, lmo, iterate, line_search=FrankWolfe.Shortstep(1), verbose=true, gradient=gradient, memory_mode=FrankWolfe.OutplaceEmphasis(), max_iteration=max_iterations, trajectory=true)\nBPFW_SS = FrankWolfe.blended_pairwise_conditional_gradient(f, grad, lmo, iterate, line_search=FrankWolfe.Shortstep(1), verbose=true, gradient=gradient, memory_mode=FrankWolfe.OutplaceEmphasis(), max_iteration=max_iterations, trajectory=true)\ndata = [FW_OL[end], FW_SS[end], BPFW_SS[end - 1]]\nlabels = [\"FW-OL\", \"FW-SS\", \"BPFW-SS\"]\nplot_trajectories(data, labels, xscalelog=true)","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"Observe that FW-OL converges with a rate of mathcalO(1t^2), which is faster than the convergence rate of mathcalO(1t) admitted by FW-SS and BPFW-SS. Explaining this phenomenon of acceleration remains an open problem.","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"","category":"page"},{"location":"examples/infinite_dimensional_kernel_herding/","page":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","title":"Kernel herding: The Frank-Wolfe algorithm in an infinite-dimensional setting","text":"This page was generated using Literate.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = KernelHerding","category":"page"},{"location":"#KernelHerding","page":"Home","title":"KernelHerding","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for KernelHerding.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [KernelHerding]","category":"page"},{"location":"#KernelHerding.KernelHerdingIterate","page":"Home","title":"KernelHerding.KernelHerdingIterate","text":"Infinite dimensional iterate for kernel herding.\n\n\n\n\n\n","category":"type"},{"location":"#KernelHerding.MeanElement","page":"Home","title":"KernelHerding.MeanElement","text":"MeanElement must implement dot with a functional.\n\n\n\n\n\n","category":"type"},{"location":"#KernelHerding.NonZeroMeanElement","page":"Home","title":"KernelHerding.NonZeroMeanElement","text":"mu =/= 0.\n\n\n\n\n\n","category":"type"},{"location":"#KernelHerding.ZeroMeanElement","page":"Home","title":"KernelHerding.ZeroMeanElement","text":"mu = 0.\n\n\n\n\n\n","category":"type"},{"location":"#Base.:*-Tuple{KernelHerdingIterate, Real}","page":"Home","title":"Base.:*","text":"Multiplication for KernelHerdingIterate.\n\n\n\n\n\n","category":"method"},{"location":"#Base.:+-Tuple{KernelHerdingIterate, KernelHerdingIterate}","page":"Home","title":"Base.:+","text":"Addition for KernelHerdingIterate.\n\n\n\n\n\n","category":"method"},{"location":"#Base.:--Tuple{KernelHerdingIterate, KernelHerdingIterate}","page":"Home","title":"Base.:-","text":"Subtraction for KernelHerdingIterate.\n\n\n\n\n\n","category":"method"},{"location":"#KernelHerding.bernoulli_polynomial-Tuple{Any}","page":"Home","title":"KernelHerding.bernoulli_polynomial","text":"The degree 2 Bernoulli polynomial.\n\n\n\n\n\n","category":"method"},{"location":"#KernelHerding.construct_rho-Tuple{Any}","page":"Home","title":"KernelHerding.construct_rho","text":"Given rho =(rhoa, rhob), returns a normalizedrho such that normalizedrho is a valid distribution.\n\n\n\n\n\n","category":"method"},{"location":"#KernelHerding.kernel_evaluation_wahba-Tuple{Any, Any}","page":"Home","title":"KernelHerding.kernel_evaluation_wahba","text":"Evaluates the Wahba kernel over two real numbers.\n\n\n\n\n\n","category":"method"},{"location":"#KernelHerding.mu_from_rho-Tuple{Any}","page":"Home","title":"KernelHerding.mu_from_rho","text":"Given a distribution ρ, computes μ.\n\n\n\n\n\n","category":"method"},{"location":"#KernelHerding.pad_non_zero_mean_element!-Tuple{KernelHerding.NonZeroMeanElement}","page":"Home","title":"KernelHerding.pad_non_zero_mean_element!","text":"The NonZeroMeanElement is represented by two vectors, cosineweights and sineweights. This function pads the shorter vector with zeros.\n\n\n\n\n\n","category":"method"},{"location":"#LinearAlgebra.dot-Tuple{KernelHerdingIterate, KernelHerdingIterate}","page":"Home","title":"LinearAlgebra.dot","text":"Scalar product for two KernelHerdingIterates.\n\n\n\n\n\n","category":"method"},{"location":"#LinearAlgebra.dot-Union{Tuple{T}, Tuple{KernelHerdingIterate{T}, KernelHerding.NonZeroMeanElement}} where T","page":"Home","title":"LinearAlgebra.dot","text":"Scalar product for KernelHerdingIterate with NonZeroMeanElement.\n\n\n\n\n\n","category":"method"},{"location":"#LinearAlgebra.dot-Union{Tuple{T}, Tuple{KernelHerdingIterate{T}, ZeroMeanElement}} where T","page":"Home","title":"LinearAlgebra.dot","text":"Scalar product for KernelHerdingIterate with ZeroMeanElement.\n\n\n\n\n\n","category":"method"},{"location":"#LinearAlgebra.norm-Tuple{KernelHerding.NonZeroMeanElement}","page":"Home","title":"LinearAlgebra.norm","text":"Norm of NonZeroMeanElement, corresponds to ||µ||.\n\n\n\n\n\n","category":"method"},{"location":"#LinearAlgebra.norm-Tuple{ZeroMeanElement}","page":"Home","title":"LinearAlgebra.norm","text":"Norm of ZeroMeanElement, corresponds to ||µ||.\n\n\n\n\n\n","category":"method"}]
}
