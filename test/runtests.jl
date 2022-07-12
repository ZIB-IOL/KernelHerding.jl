using KernelHerding
using Test




# Elementary operations 

@testset "dot" begin
    x = KernelHerdingIterate([0.2, 0.3, 0.5], [0.1, 0, 0.7])
    y = dot(x, x)
    @assert y ≈ 0.022433333333333326
    x = KernelHerdingIterate([0.5, 0.0, 0.5], [0.1, 0, 0.7])
    y = dot(x, x)
    @assert y ≈ 0.02333333333333333
    x = KernelHerdingIterate([0.5, 0.0, 0.5], [0.1, 0, 0.7])
    w = KernelHerdingIterate([0.2, 0.8], [0.5, 0.0])
    @assert dot(w, x) == dot(x, w) 
    @assert dot(w, x) ≈ 0.003333333333333327
end

@testset "Addition" begin
    x = KernelHerdingIterate([0.1, 0.2, 0.7], [0.3, 0.32, 0.35])
    y = x + x
    @test y.weights == [0.2, 0.4, 1.4]
    @test y.vertices == [0.3, 0.32, 0.35]

    w = KernelHerdingIterate([1.0], [0.2])
    z = w + x
    @test z.weights == [1.0, 0.1, 0.2, 0.7]
end

@testset "Scalar multiplication" begin
    x = KernelHerdingIterate([0.1, 0.2, 0.7], [0.3, 0.32, 0.35])
    y = x * 0.5
    @test y.weights == [0.05, 0.1, 0.35]
end

@testset "Subtraction" begin
    x = KernelHerdingIterate([0.1, 0.9], [0.3, 0.35])
    y = KernelHerdingIterate([0.2, 0.8], [0.3, 0.0])
    z = x - y
    @test z.weights == [-0.1, 0.9, -0.8]
    @test z.vertices == [0.3, 0.35, 0.0]
end



@testset "Merging" begin
    x = KernelHerdingIterate([0.1, 0.2, 0.7], [0.3, 0.32, 0.35])
    w = KernelHerdingIterate([1.0], [0.5])
    scalar = 0.5
    KernelHerding.merge_kernel_herding_iterates(x, w, scalar)
    @test x.weights == [0.05, 0.1, 0.35, 0.5]

    x = KernelHerdingIterate([0.1, 0.2, 0.7], [0.3, 0.32, 0.35])
    w = KernelHerdingIterate([1.0], [0.3])
    scalar = 0.5
    KernelHerding.merge_kernel_herding_iterates(x, w, scalar)
    @test x.weights == [0.55, 0.1, 0.35]
end

# Creating μ

@testset "Making rho a distribution and computing mu from rho" begin
rho = ([0], [1])
normalized_rho = construct_rho(rho)
@test normalized_rho[1] == [0.0]
@test normalized_rho[2] ≈ [1.41421356]
mu = mu_from_rho(normalized_rho)
@test mu.cosine_weights ≈ [0., -0.00633257397764611]
@test mu.sine_weights ≈ [0., 0.]

rho = ([1], [1])
normalized_rho = construct_rho(rho)
@test normalized_rho[1] == [1.0]
@test normalized_rho[2] ≈ [1.0]
mu = mu_from_rho(normalized_rho)
@test mu.cosine_weights ≈ [0., 0.]
@test mu.sine_weights ≈ [0., -0.00633257397764611]

rho = ([1, 0.5, 0.2], [0.0, 0.0, 0.1, 1])
normalized_rho = construct_rho(rho)
@test normalized_rho[1] ≈ [0.93250481, 0.4662524, 0.18650096]
@test normalized_rho[2] ≈ [0., 0., 0.09325048, 0.93250481]
mu = mu_from_rho(normalized_rho)
@test mu.cosine_weights ≈ [ 1.54184410e-02,  3.85461025e-03,  1.22368579e-03,  4.47410118e-04,
8.81053771e-05,  9.17764345e-06, -4.49517230e-05, -1.72080815e-04] atol=1e-5
@test mu.sine_weights ≈ [-5.50658607e-03, -3.30395164e-03, -2.44737159e-03, -1.37664652e-04,
-9.25106459e-04, -3.18158306e-04, -8.99034460e-05,  0.00000000e+00] atol=1e-5

rho = ([-1.0, 2.0], [-3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
normalized_rho = construct_rho(rho)
@test normalized_rho[1] ≈ [-0.36514837, 0.73029674]
@test normalized_rho[2] ≈ [-1.09544512, 0., 0., 0., 0., 0., 0., 0.36514837]
mu = mu_from_rho(normalized_rho)
@test mu.cosine_weights ≈ [-6.75474558e-03, -3.37737279e-03, -7.50527286e-04,  4.22171599e-04,
0.00000000e+00,  0.00000000e+00, -2.06777926e-04,  0.00000000e+00,
1.25087881e-04,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -6.59643123e-06] atol=1e-5
@test mu.sine_weights ≈ [-2.02642367e-02, -2.53302959e-03,  2.25158186e-03,  0.00000000e+00,
0.00000000e+00, -1.87631822e-04,  6.89259753e-05,  0.00000000e+00,
4.16959603e-05, -6.75474558e-05,  0.00000000e+00,  0.00000000e+00,
0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00] atol=1e-5
end

# Padding μ

@testset "Padding" begin
    mu = KernelHerding.NonZeroMeanElement([0., 0.], [1.])
    KernelHerding.pad_non_zero_mean_element!(mu)
    @test length(mu.cosine_weights) == length(mu.sine_weights) 
    mu = KernelHerding.NonZeroMeanElement([0., 0., 0.5], [1.])
    KernelHerding.pad_non_zero_mean_element!(mu)
    @test length(mu.cosine_weights) == length(mu.sine_weights) == 3
    @test mu.cosine_weights == [0., 0., 0.5]
    @test mu.sine_weights == [1., 0., 0.]
end




