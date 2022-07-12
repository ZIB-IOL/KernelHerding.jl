module KernelHerding
using FrankWolfe
using LinearAlgebra

include("iterates.jl")
export KernelHerdingIterate, KernelHerdingGradient, MarginalPolytopeWahba, mu_from_rho, construct_rho
end
