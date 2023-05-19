using GLMakie, AttractorConvergence, SparseArrays
import MarkovChainHammer.Utils: histogram
using MarkovChainHammer.BayesianMatrix
using MarkovChainHammer.TransitionMatrix: entropy
using MarkovChainHammer.TransitionMatrix: steady_state

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
# read centers list from file 
hfile = h5open(pwd() * "/data/kmeans.hdf5", "r")
centers_matrix = read(hfile["centers"])
levels = read(hfile["levels"])
close(hfile)
centers_list = [[centers_matrix[:, 1, i], centers_matrix[:, 2, i]] for i in 1:size(centers_matrix)[3]]
##
embedding = StateTreeEmbedding(centers_list, levels)
##
# Need consistency between centers and markov_chain
observable(state) = state[3]
gₜ = [observable(timeseries[:, i]) for i in 1:size(timeseries)[2]]
level_list = 1:levels
time_moment = mean(gₜ)
observable_lists = Vector{Float64}[]
observable_values = Float64[]
for level in ProgressBar(level_list)
    markov_states = get_markov_states(centers_list, level)
    gₑ = observable.(markov_states)
    p = probabilities_list[level]
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
prob_lim = extrema(probabilities_t)
prob_lim = (0, 0.2)
barplot!(ax1, values_t, probabilities_t; color=(:blue, 0.5), gap=0.0, options...)
ylims!(ax1, prob_lim)

for (i, level) in enumerate(levels-7:levels)
    ii = i ÷ 3 + 1
    jj = i % 3 + 1
    level = levels - i + 1
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
relative_error_list = Vector{Float64}[]
for level in 1:levels
    ensemble_moments = [sum(moment_functions[i].(observable_lists[level]) .* probabilities_list[level]) for i in 1:num_moments]
    push!(ensemble_moment_list, ensemble_moments)
    relative_error = [abs(ensemble_moments[i] - temporal_moments[i]) / temporal_moments[i] * 100 for i in 1:num_moments]
    push!(relative_error_list, relative_error)
end
