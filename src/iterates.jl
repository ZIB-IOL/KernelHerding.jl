

"""
Infinite dimensional iterate for kernel herding.
"""
struct KernelHerdingIterate{T}
    weights::Vector{T}
    vertices::Vector{T}
end


LinearAlgebra.dot(y, x::KernelHerdingIterate) = LinearAlgebra.dot(x, y)

"""
Addition for KernelHerdingIterate.
"""
function Base.:+(x1::KernelHerdingIterate, x2::KernelHerdingIterate)
    x = copy(x1)
    for idx2 in eachindex(x2.vertices)
        p2 = x2.vertices[idx2]
        found = false
        for idx in eachindex(x.vertices)
            p = x.vertices[idx]
            if p≈p2
                # @info "found vertex $idx2"
                # @info "vertex weights before $(x.weights)"
                x.weights[idx] += x2.weights[idx2]
                # @info "found vertex $idx"
                # @info "vertex weights after $(x.weights)"
                found = true
            end
        end
        if !found
            push!(x.weights, x2.weights[idx2])
            push!(x.vertices, x2.vertices[idx2])
        end
    end
    return x
end

"""
Multiplication for KernelHerdingIterate.
"""
function Base.:*(x::KernelHerdingIterate, scalar::Real)
    w = copy(x)
    w.weights .*= scalar
    return w
end

function Base.:*(scalar::Real, x::KernelHerdingIterate)
    w = copy(x)
    w.weights .*= scalar
    return w
end

"""
Subtraction for KernelHerdingIterate.
"""
function Base.:-(x1::KernelHerdingIterate, x2::KernelHerdingIterate)
    return x1 + (-1)*x2
end







# # Different types of distributions


"""
MeanElement must implement dot with a functional.
"""
abstract type MeanElement
end




"""
mu = 0.
"""
struct ZeroMeanElement <: MeanElement
end

"""
mu =/= 0.
"""
struct NonZeroMeanElement{T} <: MeanElement
    cosine_weights::Vector{T}
    sine_weights::Vector{T}
end


"""
Scalar product for two KernelHerdingIterates.
"""
function LinearAlgebra.dot(x1::KernelHerdingIterate, x2::KernelHerdingIterate)
    w_1 = x1.weights
    w_2 = x2.weights
    v_1 = x1.vertices
    v_2 = x2.vertices
    w_matrix = w_1 * w_2'
    p_matrix = [kernel_evaluation_wahba.(v_1[i], v_2[j]) for i in eachindex(v_1), j in eachindex(v_2)]
    scalar_product_matrices = dot(w_matrix, p_matrix)
    return scalar_product_matrices
end


"""
Scalar product for KernelHerdingIterate with ZeroMeanElement.
"""
function LinearAlgebra.dot(x::KernelHerdingIterate{T}, mu::ZeroMeanElement) where T
    return zero(T)
end



"""
Norm of ZeroMeanElement, corresponds to ||µ||.
"""
function LinearAlgebra.norm(mu::ZeroMeanElement)
    return zero(Float64)
end

"""
The NonZeroMeanElement is represented by two vectors, cosine_weights and sine_weights. This function pads the shorter vector with zeros.
"""
function pad_non_zero_mean_element!(mu::NonZeroMeanElement)
    mu_a = mu.cosine_weights
    mu_b = mu.sine_weights
    if length(mu_a) > length(mu_b)
        for _ in length(mu_b) + 1: length(mu_a)
            push!(mu_b, 0)
        end
    elseif length(mu_a) < length(mu_b)
        for _ in length(mu_a) + 1: length(mu_b)
            push!(mu_a, 0)
        end
    end
end


"""
Scalar product for KernelHerdingIterate with NonZeroMeanElement.
"""
function LinearAlgebra.dot(x::KernelHerdingIterate{T}, mu::NonZeroMeanElement) where T
    pad_non_zero_mean_element!(mu)
    mu_a = mu.cosine_weights
    mu_b = mu.sine_weights
    weights = x.weights
    vertices = x.vertices
    value = 0
    for j in eachindex(weights)
        value_j = 0
        for i in eachindex(mu.cosine_weights)
            value_j += mu_a[i] * cos(2π * i * vertices[j]) + mu_b[i] * sin(2π * i * vertices[j])
        end
        value += weights[j]*value_j
    end
    return value
end

"""
Norm of NonZeroMeanElement, corresponds to ||µ||.
"""
function LinearAlgebra.norm(mu::NonZeroMeanElement)
    
    pad_non_zero_mean_element!(mu)
    mu_a = mu.cosine_weights
    mu_b = mu.sine_weights
    
    mu_squared = sum(eachindex(mu_a)) do i
        (2π * i)^2 * (mu_a[i]^2 + mu_b[i]^2)
    end
    mu_squared /= 2
    return sqrt(mu_squared)
end



# # Technical replacements for the infinite-dimensional kernel herding setting.

function Base.similar(x::KernelHerdingIterate, ::Type{T}) where T
    return KernelHerdingIterate(similar(x.weights, T), similar(x.vertices, T))
end

function Base.similar(x::KernelHerdingIterate{T}) where T
    return Base.similar(x, T)
end

function Base.eltype(x::KernelHerdingIterate{T}) where T
    return T
end

function FrankWolfe.compute_active_set_iterate!(active_set::FrankWolfe.ActiveSet{AT,R,IT}) where {AT, R, IT <: KernelHerdingIterate}

    empty!(active_set.x.weights)
    empty!(active_set.x.vertices)

    for idx in eachindex(active_set)
        push!(active_set.x.weights, active_set.weights[idx])
        push!(active_set.x.vertices, only(active_set.atoms[idx].vertices))
    end
end

function Base.empty!(active_set::FrankWolfe.ActiveSet{AT,R,IT}) where {AT, R, IT <: KernelHerdingIterate}
    empty!(active_set.atoms)
    empty!(active_set.weights)
    empty!(active_set.x.weights)
    empty!(active_set.x.vertices)
    push!(active_set.x.weights, 1)
    push!(active_set.x.vertices, 0)
    return active_set
end


function FrankWolfe.active_set_initialize!(active_set::FrankWolfe.ActiveSet{AT,R,IT}, v::KernelHerdingIterate) where {AT, R, IT <: KernelHerdingIterate}
    empty!(active_set)
    push!(active_set, (one(R), v))
    FrankWolfe.compute_active_set_iterate!(active_set)
    return active_set
end


function FrankWolfe.active_set_update!(active_set::FrankWolfe.ActiveSet{AT, R, IT}, lambda, atom, renorm=true, idx=nothing) where {AT, R, IT <: KernelHerdingIterate}
    
    # rescale active set
    active_set.weights .*= (1 - lambda)
    # add value for new atom
    if idx === nothing
        idx = FrankWolfe.find_atom(active_set, atom)
    end
    updating = false
    if idx > 0
        @inbounds active_set.weights[idx] = active_set.weights[idx] + lambda
        updating = true
    else
        push!(active_set, (lambda, atom))
    end
    if renorm
        FrankWolfe.active_set_cleanup!(active_set, update=false)
        FrankWolfe.active_set_renormalize!(active_set)
    end
    merge_kernel_herding_iterates(active_set.x, atom, lambda)
    return active_set
end



function FrankWolfe.active_set_update_iterate_pairwise!(active_set::FrankWolfe.ActiveSet{AT, R, IT}, lambda, fw_atom, away_atom) where {AT, R, IT <: KernelHerdingIterate}
    x = active_set.x + lambda * fw_atom - lambda * away_atom
    copy!(active_set.x, x)
    return active_set
end




function merge_kernel_herding_iterates(x1::KernelHerdingIterate, x2::KernelHerdingIterate, scalar)
    @assert(0 <= scalar <= 1, "Scalar should be in [0, 1].")
    if scalar == 0
        return x1
    elseif scalar == 1
        return x2
    else
        x1.weights .*= (1 - scalar)
        for idx in eachindex(x2.weights)
            p_2 = x2.vertices[idx]
            found = false
            for idx2 in eachindex(x1.weights)
                p_1 = x1.vertices[idx2]
                if p_2 ≈ p_1
                    x1.weights[idx2] += x2.weights[idx] * scalar
                    found = true
                end
            end
            if !found
                push!(x1.weights, x2.weights[idx]*scalar)
                push!(x1.vertices, x2.vertices[idx])
            end
        end
    end
# @assert sum(x1.weights) ≈ 1
@assert all(0 .<= x1.vertices .<=1)
end

function Base.copy(x::KernelHerdingIterate{T}) where T
    return KernelHerdingIterate(copy(x.weights), copy(x.vertices))
end

function Base.copy!(x::KernelHerdingIterate{T}, y::KernelHerdingIterate{T}) where T
    copy!(x.weights, y.weights)
    copy!(x.vertices, y.vertices)
end


mutable struct KernelHerdingGradient{T, D <: MeanElement}
    x:: KernelHerdingIterate{T}
    mu:: D
end

function LinearAlgebra.dot(x::KernelHerdingIterate, g::KernelHerdingGradient)
    scalar_product = dot(g.x, x)
    scalar_product -= dot(g.mu, x)
    return scalar_product
end

function create_loss_function_gradient(mu::MeanElement)
    
    mu_squared = norm(mu)^2
    function evaluate_loss(x::KernelHerdingIterate)
        l = dot(x, x)
        l += mu_squared
        l += dot(x, mu)
        l *= 1 / 2
        return l
    end
    function evaluate_gradient(g::KernelHerdingGradient, x::KernelHerdingIterate)
        g.x = x
        g.mu = mu
    return g
    end
    return evaluate_loss, evaluate_gradient
end


struct MarginalPolytopeWahba <: FrankWolfe.LinearMinimizationOracle
    number_iterations:: Int
end

function FrankWolfe.compute_extreme_point(lmo::MarginalPolytopeWahba, direction::KernelHerdingGradient; kw...)
    
    optimal_value = Inf
    optimal_vertex = nothing
    current_vertex = nothing
    for iteration in 0:lmo.number_iterations - 1
        current_vertex = KernelHerdingIterate([1.0], [iteration / (lmo.number_iterations)])
        current_value = dot(direction, current_vertex)
        if current_value < optimal_value
            optimal_vertex = current_vertex
            optimal_value = current_value
        end
    end
    @assert(optimal_vertex !== nothing, "This should never happen.")
    @assert(0 <= only(optimal_vertex.vertices) <= 1, "Vertices have to correspond to real numbers in [0, 1].")
    return optimal_vertex
end


"""
The degree 2 Bernoulli polynomial.
"""
function bernoulli_polynomial(y)
    return (1/2) * (y^2 - y + 1/6)
end


"""
Evaluates the Wahba kernel over two real numbers.
"""
function kernel_evaluation_wahba(y1, y2)
    y = y1 - y2
    return bernoulli_polynomial(y - floor(y))
end


"""
Given rho =(rho_a, rho_b), returns a normalized_rho such that normalized_rho is a valid distribution.
"""
function construct_rho(rho)
    rho_a = rho[1]
    rho_b = rho[2]
    summed_up = 1/2 * (norm(rho_a)^2 + norm(rho_b)^2)
    rho_a = rho_a / sqrt(summed_up)
    rho_b = rho_b / sqrt(summed_up)
    summed_up = 1/2 * (norm(rho_a)^2 + norm(rho_b)^2)
    @assert(summed_up ≈ 1, "The tuple ρ still does not correspond to a distribution.")
    normalized_rho = (rho_a, rho_b)
    return normalized_rho
end

"""
Given a distribution ρ, computes μ.
"""
function mu_from_rho(rho)
    rho_a = rho[1]
    rho_b = rho[2]
    summed_up = 1/2 * (norm(rho_a)^2 + norm(rho_b)^2)
    @assert(summed_up ≈ 1, "The tuple ρ still does not correspond to a distribution.")
    length_a = length(rho_a)
    length_b = length(rho_b)
    length_max = 2 * max(length_a, length_b)

    mu_a = zeros(length_max)
    mu_b = zeros(length_max)
    for i in eachindex(rho_a)
        for j in eachindex(rho_a)
            sum_i_j = i + j
            mu_a[sum_i_j] += 1 / 4 * rho_a[i] * rho_a[j] * 2 / ((2π * sum_i_j)^2)
            diff_i_j = abs(i - j)
            if diff_i_j > 0
                mu_a[diff_i_j] += 1 / 4 * rho_a[i] * rho_a[j] * 2 / ((2π * diff_i_j)^2)
            end
        end
        for k in eachindex(rho_b)
            sum_i_k = i + k
            mu_b[sum_i_k] -= 2 / 4 * rho_a[i] * rho_b[k] * 2 / ((2π * sum_i_k)^2)
            diff_i_k = abs(i - k)
            if diff_i_k > 0
                val = 2 / 4 * rho_a[i] * rho_b[k] * 2 / ((2π * diff_i_k)^2)
                if i > k
                    mu_b[diff_i_k] += val
                elseif k > i 
                    mu_b[diff_i_k] -= val
                end
            end
        end
    end
    for i in eachindex(rho_b)
        for j in eachindex(rho_b)
            sum_i_j = i + j
            mu_a[sum_i_j] -= 1 / 4 * rho_b[i] * rho_b[j] * 2 / ((2π * sum_i_j)^2)
            diff_i_j = abs(i - j)
            if diff_i_j > 0
                mu_a[diff_i_j] += 1 / 4 * rho_b[i] * rho_b[j] * 2 / ((2π * diff_i_j)^2)
            end
        end
    end
    mu = NonZeroMeanElement(mu_a, mu_b)
    return mu
end