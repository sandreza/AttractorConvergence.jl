module AttractorConvergence

using LinearAlgebra, SparseArrays, Random, Statistics
using ProgressBars

export lorenz!, lorenz, lorenz_data, lorenz_symmetry
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

function lorenz_data(; timesteps=10^7, Δt=0.005, ϵ=0.0, ρ=t -> 28.0, initial_condition=[1.4237717232359446, 1.778970017190979, 16.738782836244038])
    rhs(x, t) = lorenz(x, ρ(t), 10.0, 8.0 / 3.0)
    x_f = zeros(3, timesteps)
    x_f[:, 1] .= initial_condition
    evolve! = RungeKutta4(3)
    for i in ProgressBar(2:timesteps)
        xOld = x_f[:, i-1]
        evolve!(rhs, xOld, Δt)
        if ϵ > 0.0
            𝒩 = randn(3)
            @inbounds @. x_f[:, i] = evolve!.xⁿ⁺¹ + ϵ * sqrt(Δt) * 𝒩
        else
            @inbounds @. x_f[:, i] = evolve!.xⁿ⁺¹
        end
    end
    return x_f, Δt
end
function lorenz_symmetry(timeseries)
    symmetrized_timeseries = zeros(size(timeseries))
    for i in ProgressBar(1:size(timeseries)[2])
        symmetrized_timeseries[1, i] = -timeseries[1, i]
        symmetrized_timeseries[2, i] = -timeseries[2, i]
        symmetrized_timeseries[3, i] = timeseries[3, i]
    end
    return symmetrized_timeseries
end

function distance_matrix(data)
    d_mat = zeros(size(data)[2], size(data)[2])
    for j in ProgressBar(1:size(data)[2])
        Threads.@threads for i in 1:j-1
            @inbounds d_mat[i, j] = norm(data[:, i] - data[:, j])
        end
    end
    return Symmetric(d_mat)
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

function sparsify(Q; threshold=eps(10^6.0))
    Q[abs.(Q).<threshold] .= 0
    return sparse(Q)
end

include("tree_structure.jl")
include("tree_embedding.jl")

end # module AttractorConvergence
