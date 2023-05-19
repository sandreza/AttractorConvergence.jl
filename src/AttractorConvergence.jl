module AttractorConvergence

using LinearAlgebra, SparseArrays
using ProgressBars

export lorenz!, lorenz
# generate data
function lorenz!(ṡ, s)
    ṡ[1] = 10.0 * (s[2] - s[1])
    ṡ[2] = s[1] * (28.0 - s[3]) - s[2]
    ṡ[3] = s[1] * s[2] - (8 / 3) * s[3]
    return nothing
end

function lorenz(x, ρ, σ, β)
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    return [σ * (x2 - x1), x1 * (ρ - x3) - x2, x1 * x2 - β * x3]
end

export RungeKutta4, rk4

function rk4(f, s, dt)
    ls = length(s)
    k1 = zeros(ls)
    k2 = zeros(ls)
    k3 = zeros(ls)
    k4 = zeros(ls)
    f(k1, s)
    f(k2, s + k1 * dt / 2)
    f(k3, s + k2 * dt / 2)
    f(k4, s + k3 * dt)
    return s + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
end
struct RungeKutta4{S,T,U}
    k⃗::S
    x̃::T
    xⁿ⁺¹::T
    t::U
end
RungeKutta4(n) = RungeKutta4(zeros(n, 4), zeros(n), zeros(n), [0.0])

function (step::RungeKutta4)(f, x, dt)
    @inbounds let
        @. step.x̃ = x
        step.k⃗[:, 1] .= f(step.x̃, step.t[1])
        @. step.x̃ = x + step.k⃗[:, 1] * dt / 2
        @. step.t += dt / 2
        step.k⃗[:, 2] .= f(step.x̃, step.t[1])
        @. step.x̃ = x + step.k⃗[:, 2] * dt / 2
        step.k⃗[:, 3] .= f(step.x̃, step.t[1])
        @. step.x̃ = x + step.k⃗[:, 3] * dt
        @. step.t += dt / 2
        step.k⃗[:, 4] .= f(step.x̃, step.t[1])
        @. step.xⁿ⁺¹ = x + (step.k⃗[:, 1] + 2 * step.k⃗[:, 2] + 2 * step.k⃗[:, 3] + step.k⃗[:, 4]) * dt / 6
    end
    return nothing
end

export StateEmbedding, StateTreeEmbedding, level_global_indices, parent_global_index, get_markov_states
export sparsify

abstract type AbstractEmbedding end

struct StateEmbedding{S} <: AbstractEmbedding
    markov_states::S
end
function (embedding::StateEmbedding)(current_state)
    argmin([norm(current_state - markov_state) for markov_state in embedding.markov_states])
end

struct StateTreeEmbedding{S, T} <: AbstractEmbedding
    markov_states::S
    levels::T
end
function (embedding::StateTreeEmbedding)(current_state)
    global_index = 1 
    for level in 1:embedding.levels
        new_index = argmin([norm(current_state - markov_state) for markov_state in embedding.markov_states[global_index]])
        global_index = child_global_index(new_index, global_index)
    end
    return local_index(global_index, embedding.levels)
end

# assumes binary tree
local_index(global_index, levels) = global_index - 2^levels + 1 # markov index from [1, 2^levels]
# parent local index is markov_index(global_index, levels-1)
# child local index is 2*markov_index(global_index, levels-1) + new_index - 1
# global index is 2^levels + 1 + child local index
child_global_index(new_index, global_parent_index, level) = (2 * (local_index(global_parent_index, level - 1)-1) + new_index - 1) + 2^(level) 
# simplified:
child_global_index(new_index, global_parent_index) = 2 * global_parent_index + new_index - 1 
# global_indices per level
level_global_indices(level) = 2^(level-1):2^level-1
parent_global_index(child_index) = div(child_index, 2) # both global

# markov states from centers list 
function get_markov_states(centers_list::Vector{Vector{Vector{Float64}}}, level)
    markov_states = Vector{Float64}[]
    indices = level_global_indices(level)
    for index in indices
        push!(markov_states, centers_list[index][1])
        push!(markov_states, centers_list[index][2])
    end
    return markov_states
end
get_markov_states(embedding::StateTreeEmbedding, level) = get_markov_states(embedding.markov_states, level)

function sparsify(Q; threshold=eps(10^6.0))
    Q[abs.(Q).<threshold] .= 0
    return sparse(Q)
end

end # module AttractorConvergence
