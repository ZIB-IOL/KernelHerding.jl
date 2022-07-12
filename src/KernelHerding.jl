module KernelHerding
using FrankWolfe
using LinearAlgebra

include("iterates.jl")
export KernelHerdingIterate, KernelHerdingGradient, MarginalPolytopeWahba, mu_from_rho, construct_rho, create_loss_function_gradient
end
