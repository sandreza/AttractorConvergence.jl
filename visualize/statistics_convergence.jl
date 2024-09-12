using CairoMakie, AttractorConvergence, SparseArrays
import MarkovChainHammer.Utils: histogram
using MarkovChainHammer.BayesianMatrix
using MarkovChainHammer.TransitionMatrix: entropy
using MarkovChainHammer.TransitionMatrix: steady_state

##
embedding = StateTreeEmbedding(centers_list, levels)
##
# Need consistency between centers and markov_chain
observable(state) = state[3]
gₜ = [observable(m_timeseries[:, i]) for i in 1:size(m_timeseries)[2]]
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

##
# observables_list = [t -> t[3], t -> t[1] * t[2], t -> t[1]^2, t -> t[2]^2, t -> t[1] * t[2] * t[3], t -> t[3] * log(t[3])]
# observables_list_names = ["z", "xy", "x²", "y²", "xyz", "z log(z)"]
observables_list = [t -> t[3], t -> t[3]^2, t -> t[3]^3, t -> t[3]^4, t -> t[3]^5, t -> t[3] * log(abs(t[3]) + eps(1.0))]
observables_list_colors = [:blue, :red, :green, :orange, :purple, :yellow]
observables_list_names = ["z", "z²", "z³", "z⁴", "z⁵", "z log(|z|)"]

observable_lists = Vector{Float64}[]
observable_values = Float64[]
observable_values_timeaverage = Float64[]

level_list = 1:levels

for observable in ProgressBar(observables_list)
    g1 = [observable(m_timeseries[:, i]) for i in 1:size(m_timeseries)[2]]
    g2 = [observable(s_timeseries[:, i]) for i in 1:size(s_timeseries)[2]]
    time_moment = mean(hcat(g1,g2))
    push!(observable_values_timeaverage, time_moment)
    for level in ProgressBar(level_list)
        markov_states = get_markov_states(centers_list, level)
        gₑ = observable.(markov_states)
        p = probabilities_list[level]
        push!(observable_lists, gₑ)
        observable_value = sum(gₑ .* p)
        push!(observable_values, observable_value)
    end
end
##
relative_error_matrix = zeros(levels, length(observables_list))
for i in eachindex(observables_list)
    local_inds = levels*(i - 1)+1:levels*i
    observable_value = observable_values[local_inds]
    relative_error_matrix[:, i] .= abs.(observable_value .- observable_values_timeaverage[i]) ./ observable_values_timeaverage[i] 
end
##
fig2 = Figure(resolution = (500, 500))
ax = Axis(fig2[1, 1]; title="Convergence", xlabel = "log10(partitions)", ylabel = "log10(relative error)")
for i in eachindex(observables_list)
    scatter!(ax, log10.(2 .^ level_list), log10.(relative_error_matrix[:, i]); color=observables_list_colors[i], label=observables_list_names[i])
end
lines!(ax, log10.(2 .^ level_list), -log10.(2 .^ level_list) .+ 0, color=:black, linestyle=:dash, linewidth=3, label="slope -1")
axislegend(ax)
display(fig2)