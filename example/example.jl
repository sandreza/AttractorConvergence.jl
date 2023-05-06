using ParallelKMeans
using GLMakie
using MarkovChainHammer, AttractorConvergence
using ProgressBars, LinearAlgebra, Statistics, Random
using MarkovChainHammer.BayesianMatrix

import MarkovChainHammer.TransitionMatrix: generator, holding_times, steady_state, perron_frobenius
import MarkovChainHammer.Utils: histogram, autocovariance

@info "evolving lorenz equations"
initial_condition = [14.0, 20.0, 27.0]
dt = 0.005
iterations = 2 * 10^5

timeseries = zeros(3, iterations)
markov_chain = zeros(Int, iterations)
timeseries[:, 1] .= initial_condition

for i in ProgressBar(2:iterations)
    # take one timestep forward via Runge-Kutta 4
    state = rk4(lorenz!, timeseries[:, i-1], dt)
    timeseries[:, i] .= state
end
##
@info "starting k-means"
X = timeseries[:,1:10:end]
numstates = 2000
r = kmeans(X, numstates; max_iters=10000)
@info "done with k-means"

##
@info "computing markov embedding"
markov_states = [r.centers[:, i] for i in 1:numstates]
# markov_states = [timeseries[:, i] for i in 1:floor(Int, iterations/numstates):iterations]
embedding = StateEmbedding(markov_states)

for i in ProgressBar(1:iterations)
    state = timeseries[:, i]
    # partition state space according to most similar markov state
    # This will be sped up by using a tree structure
    markov_index = embedding(state)
    markov_chain[i] = markov_index
end
##
@info "constructing the generator"
Q = mean(BayesianGenerator(markov_chain; dt=dt))
p = steady_state(Q)
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
indexend = minimum([1000000, iterations])
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