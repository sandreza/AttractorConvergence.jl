using GLMakie, AttractorConvergence
import MarkovChainHammer.Utils: histogram
using MarkovChainHammer.BayesianMatrix

@info "loading data"
hfile = h5open(pwd() * "/data/embedding.hdf5", "r")
markov_chain = read(hfile["markov_chain"])
s_markov_chain = read(hfile["symmetrized markov chain"])
Δt = read(hfile["dt"])
close(hfile)
hfile = h5open(pwd() * "/data/lorenz.hdf5", "r")
timeseries = read(hfile["timeseries"])
s_timeseries = read(hfile["symmetrized timeseries"])
close(hfile)
##
embedding = StateTreeEmbedding(centers_list, levels)
##
function get_markov_states(centers_list::Vector{Vector{Vector{Float64}}}, level)
    markov_states = Vector{Float64}[]
    indices = level_global_indices(level)
    for index in indices
        push!(markov_states, centers_list[index][1])
        push!(markov_states, centers_list[index][2])
    end
    return markov_states
end
# 1 + 
##
# Need consistency between centers and markov_chain
observable(state) = state[3]
gₜ = [observable(timeseries[:, i]) for i in 1:size(timeseries)[2]]
level_list = 1:10
time_moment = mean(gₜ)
observable_lists = Vector{Float64}[]
probabilities_list = Vector{Float64}[]
observable_values = Float64[]
for level in ProgressBar(level_list)
    markov_states = get_markov_states(centers_list, level)
    coarse_grained_markov_chain = div.(markov_chain .- 1, 2^(10 - level)) .+ 1
    Q = mean(BayesianGenerator(coarse_grained_markov_chain; dt=Δt))
    p = steady_state(Q)
    push!(probabilities_list, p)
    gₑ = observable.(markov_states)
    push!(observable_lists, gₑ)
    observable_value = sum(gₑ .* p)
    push!(observable_values, observable_value)
end
##
fig = Figure(resolution=(1500, 1500))
ax1 = Axis(fig[1, 1]; title="timeseries")
labelsize = 40
options = (; xlabel="Time [days]", ylabel="Probability", titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
custom_range = extrema(gₜ)
values_t, probabilities_t = histogram(gₜ; bins=20, custom_range=custom_range)
prob_lim = extrema(probabilities)
prob_lim = (0, 0.2)
barplot!(ax1, values_t, probabilities_t; color=(:blue, 0.5), gap=0.0, options...)
ylims!(ax1, prob_lim)

for level in 3:10
    ii = (level - 2) ÷ 3 + 1
    jj = (level - 2) % 3 + 1
    level = 13 - level
    gₑ = observable_lists[level]
    p = probabilities_list[level]
    ax2 = Axis(fig[ii, jj]; title="ensemble level $level")
    values, probabilities = histogram(gₑ; bins=20, normalization=p, custom_range=custom_range)
    barplot!(ax2, values, probabilities; color=(:red, 0.5), gap=0.0, options...)
    ylims!(ax2, prob_lim)
end
display(fig)

##
moment_functions = [t -> t^i for i in 1:10]
num_moments = 6
temporal_moments = [mean(moment_functions[i].(gₜ)) for i in 1:num_moments]
ensemble_moment_list = Vector{Float64}[]
for level in 1:10
    ensemble_moments = [sum(moment_functions[i].(observable_lists[level]) .* probabilities_list[level]) for i in 1:num_moments]
    push!(ensemble_moment_list, ensemble_moments)
end