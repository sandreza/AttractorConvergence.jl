using ParallelKMeans
using CairoMakie, Revise
using MarkovChainHammer, AttractorConvergence
using ProgressBars, LinearAlgebra, Statistics, Random
using MarkovChainHammer.BayesianMatrix
import MarkovChainHammer.TransitionMatrix: generator, holding_times, steady_state, perron_frobenius, entropy
import MarkovChainHammer.Utils: histogram, autocovariance
Random.seed!(12345)

@info "evolving lorenz equations"
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
            @inbounds d_mat[i,j] = norm(data[:, i] - data[:, j])
        end
    end
    return Symmetric(d_mat)
end
timesteps = 10^7
timeseries, Δt = lorenz_data(timesteps=timesteps, Δt=0.005)
s_timeseries = lorenz_symmetry(timeseries)
joined_timeseries = hcat(timeseries, s_timeseries) # only for Partitioning Purpose
##
@info "starting k-means"
X = joined_timeseries[:,1:1:end]
##
function split(X)
    numstates = 2
    r0 = kmeans(X, numstates; max_iters=10000)
    child_0 = (r0.assignments .== 1)
    child_1 = (!).(child_0)
    children = [view(X, :, child_0), view(X, :, child_1)]
    return r0.centers, children
end
level_global_indices(level) = 2^(level-1):2^level-1
##
levels = 10
parent_views = []
centers_list = []
push!(parent_views, X)
## Level 1
centers, children = split(X)
push!(centers_list, [centers[:, 1], centers[:, 2]])
push!(parent_views, children[1])
push!(parent_views, children[2])
## Levels 2 through levels
for level in ProgressBar(2:levels)
    for parent_global_index in level_global_indices(level)
        centers, children = split(parent_views[parent_global_index])
        push!(centers_list, [centers[:, 1], centers[:, 2]])
        push!(parent_views, children[1])
        push!(parent_views, children[2])
    end
end
@info "done with k-means"
##
# constructing embedding with 2^levels number of states
embedding = StateTreeEmbedding(centers_list, levels)
##
@info "computing markov embedding"
markov_chain = zeros(Int64, size(timeseries)[2])
for i in ProgressBar(1:size(timeseries)[2])
    state = timeseries[:, i]
    # partition state space according to most similar markov state
    # This will be sped up by using a tree structure
    markov_i = embedding(state)
    @inbounds markov_chain[i] = markov_i
end
@info "computing symmetric embedding"
s_markov_chain = zeros(Int64, size(s_timeseries)[2])
for i in ProgressBar(1:size(s_timeseries)[2])
    state = s_timeseries[:, i]
    # partition state space according to most similar markov state
    # This will be sped up by using a tree structure
    markov_i = embedding(state)
    @inbounds s_markov_chain[i] = markov_i
end
##
@info "constructing the generator"
Q = BayesianGenerator(markov_chain; dt= Δt)
Qb = BayesianGenerator(s_markov_chain, Q.posterior; dt=Δt)
Q = mean(Qb)
p = steady_state(Q)
entropy(p)
Q̃ = Diagonal(1 ./ sqrt.(p)) * Q * Diagonal(sqrt.(p))
Q̃ₛ = Symmetric((Q̃ + Q̃') / 2)
Q̃ₐ = (Q̃ - Q̃') / 2
##
Λ, V = eigen(Q)
W = inv(V)
##
koopman_mode_1 = real(W[end-1, :])
koopman_mode_2 = imag(W[end-2, :])
koopman_mode_3 = real(W[end-3, :])
koopman_mode_4 = real(W[end-8, :])
transfer_mode_1 = real(V[:, end])
transfer_mode_2 = real(V[:, end-3])
transfer_mode_3 = real(V[end-1, :])
transfer_mode_4 = imag(V[end-2, :])
predictability_index = sum(Q .> eps(10.0^8), dims=1)[:]
modes = [koopman_mode_1, koopman_mode_2, koopman_mode_3, koopman_mode_4]
modes_tr = [transfer_mode_1, transfer_mode_2, transfer_mode_3, transfer_mode_4]
##
subsampling = 1
indexstart = 1
indexend = minimum([1000000, timesteps])
indices = indexstart:subsampling:indexend
subsampled_timeseries = timeseries[:, indices]
colors = Vector{Float64}[]
for mode in modes
    color = [mode[markov_chain[i]] for i in indices]
    push!(colors, color)
end
colors_koopman = copy(colors)
##
@info "plotting"
set_theme!(backgroundcolor=:black)
fig = Figure()
ax11 = LScene(fig[1, 1]; show_axis=false)
ax12 = LScene(fig[1, 2]; show_axis=false)
ax21 = LScene(fig[2, 1]; show_axis=false)
ax22 = LScene(fig[2, 2]; show_axis=false)
ax = [ax11, ax12, ax21, ax22]
for i in 1:4
    color = colors_koopman[i]
    colorrange = (-maximum(color), maximum(color))
    lines!(ax[i], subsampled_timeseries[:, :], color=color, colormap=:balance, colorrange=colorrange, linewidth=1)
    rotate_cam!(ax[i].scene, (0, 11, 0))
end
display(fig)
##
fig2 = Figure(resolution = (2000,2000))
set_theme!(backgroundcolor=:black)
for i in 1:9
    ii = (i - 1) ÷ 3 + 1
    jj = (i - 1) % 3 + 1
    ax = LScene(fig2[ii, jj]; show_axis=false)
    markov_indices = div.(markov_chain[inds] .+ 2^levels .- 1, 2^(levels - i))
    scatter!(ax, timeseries[:, inds], color=markov_indices, colormap=:glasbey_hv_n256, markersize=3)
    rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
end
display(fig2)