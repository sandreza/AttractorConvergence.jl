@info "opening centers"
centerslist = []
hfile = h5open(pwd() * data_directory  * "/centers.hdf5", "r")
for i in ProgressBar(eachindex(coarse_probabilities))
    centers = read(hfile["centers $i"])
    push!(centerslist, centers)
end
close(hfile)

@info "loading data"
hfile = h5open(pwd() * data_directory  * "/lorenz.hdf5", "r")
m_timeseries = read(hfile["timeseries"])
s_timeseries = read(hfile["symmetrized timeseries"])
joined_timeseries = hcat(m_timeseries, s_timeseries) # only for Partitioning Purpose
close(hfile)

@info "grabbing embedding"
hfile = h5open(pwd() * data_directory  * "/embedding.hdf5", "r")
markov_chain = read(hfile["markov_chain"])
coarse_markov_chain = read(hfile["coarse_markov_chains"])
probability = read(hfile["probability"])
coarse_probabilities = read(hfile["coarse_probabilities"])
close(hfile)

# compute transition matrix (should use only half the data)
@info "computing transition matrices"
Ps = []
N = size(coarse_markov_chain)[1]
N2 = N ÷ 2
for i in ProgressBar(1:14)
    P1 = perron_frobenius(coarse_markov_chain[1:N2, i])
    P2 = perron_frobenius(coarse_markov_chain[N2+1:end, i])
    P = (P1 + P2)/2
    push!(Ps, P)
end

##
observables = [i -> i[3]^j for j in 1:5]
observables = [observables..., i -> (i[3] .* log(i[3]))]
@info "computing observables from timeseries"
observables_list = zeros(length(observables))
for j in ProgressBar(eachindex(observables))
    N = length(eachcol(joined_timeseries))
    for state in ProgressBar(eachcol(joined_timeseries))
        observables_list[j] += observables[j](state) / N
    end
end

##
@info "computing observables from model"
Npartitions = 14
observables_list_model = zeros(Npartitions, length(observables))
for i in ProgressBar(1:Npartitions)
    p = MarkovChainHammer.Utils.steady_state(Ps[i])
    for j in eachindex(p)
        for k in eachindex(observables)
            observables_list_model[i, k] += observables[k](centerslist[i][:, j]) .* p[j]
        end
    end
end
@info "plotting observables"
##

log10errors = log10.(abs.(observables_list_model .- reshape(observables_list, (1, length(observables)))))

colors = [:red, :green, :blue, :orange, :purple, :cyan, :magenta, :black, :white]
fig = Figure()
ax = Axis(fig[1, 1])
log10partition_numbers = log10.([maximum(coarse_markov_chain[:, i]) for i in 1:Npartitions])
for i in eachindex(observables)
    scatter!(ax, log10partition_numbers, log10errors[:, i], color = (colors[i]))
end
lines!(ax, log10partition_numbers, - log10partition_numbers .+ 1, color = :black, linestyle=:dash)
display(fig)

##
@info "calculating cumulants"
# Cumulants 
cumulants_list = similar(observables_list)
cumulants_list[1] = observables_list[1]
cumulants_list[2] = observables_list[2] - observables_list[1]^2
cumulants_list[3] = observables_list[3] - 3 * observables_list[1] * observables_list[2] + 2 * observables_list[1]^3
cumulants_list[4] = observables_list[4] - 4 * observables_list[1] * observables_list[3] - 3 * observables_list[2]^2 + 12 * observables_list[1]^2 * observables_list[2] - 6 * observables_list[1]^4
cumulants_list[5] = observables_list[5] - 5 * observables_list[1] * observables_list[4] - 10 * observables_list[2] * observables_list[3] + 20 * observables_list[1]^2 * observables_list[3] + 30 * observables_list[1] * observables_list[2]^2 - 60 * observables_list[1]^3 * observables_list[2] + 24 * observables_list[1]^5
cumulants_list[6] = observables_list[6] - 6 * observables_list[1] * observables_list[5] - 15 * observables_list[2] * observables_list[4] - 10 * observables_list[3]^2 + 60 * observables_list[1]^2 * observables_list[4] + 90 * observables_list[1] * observables_list[2] * observables_list[3] - 120 * observables_list[1]^3 * observables_list[3] - 120 * observables_list[1]^2 * observables_list[2]^2 + 210 * observables_list[1]^4 * observables_list[2] - 120 * observables_list[1]^6

cumulants_list_model = similar(observables_list_model)
cumulants_list_model[:, 1] = observables_list_model[:, 1]
cumulants_list_model[:, 2] = observables_list_model[:, 2] - observables_list_model[:, 1].^2
cumulants_list_model[:, 3] = observables_list_model[:, 3] - 3 * observables_list_model[:, 1] .* observables_list_model[:, 2] + 2 * observables_list_model[:, 1].^3
cumulants_list_model[:, 4] = observables_list_model[:, 4] - 4 * observables_list_model[:, 1] .* observables_list_model[:, 3] - 3 * observables_list_model[:, 2].^2 + 12 * observables_list_model[:, 1].^2 .* observables_list_model[:, 2] - 6 * observables_list_model[:, 1].^4
cumulants_list_model[:, 5] = observables_list_model[:, 5] - 5 * observables_list_model[:, 1] .* observables_list_model[:, 4] - 10 * observables_list_model[:, 2] .* observables_list_model[:, 3] + 20 * observables_list_model[:, 1].^2 .* observables_list_model[:, 3] + 30 * observables_list_model[:, 1] .* observables_list_model[:, 2].^2 - 60 * observables_list_model[:, 1].^3 .* observables_list_model[:, 2] + 24 * observables_list_model[:, 1].^5
cumulants_list_model[:, 6] = observables_list_model[:, 6] - 6 * observables_list_model[:, 1] .* observables_list_model[:, 5] - 15 * observables_list_model[:, 2] .* observables_list_model[:, 4] - 10 * observables_list_model[:, 3].^2 + 60 * observables_list_model[:, 1].^2 .* observables_list_model[:, 4] + 90 * observables_list_model[:, 1] .* observables_list_model[:, 2] .* observables_list_model[:, 3] - 120 * observables_list_model[:, 1].^3 .* observables_list_model[:, 3] - 120 * observables_list_model[:, 1].^2 .* observables_list_model[:, 2].^2 + 210 * observables_list_model[:, 1].^4 .* observables_list_model[:, 2] - 120 * observables_list_model[:, 1].^6

##
@info "plotting cumulantsi"
ms = 20
ls = 40
lw = 5
labels = ["κ₁", "κ₂", "κ₃", "κ₄", "κ₅", "κ₆"]
log10cumulantserror = log10.(abs.(cumulants_list_model .- reshape(cumulants_list, (1, length(observables)))))
fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "log10(partitions)", ylabel = "log10(error)", xlabelsize = ls, ylabelsize = ls, xticklabelsize = ls, yticklabelsize = ls)
for i in 1:5
    scatter!(ax, log10partition_numbers, log10cumulantserror[:, i], color = (colors[i]), markersize = 20, label = labels[i])
end
axislegend(ax, position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)
lines!(ax, log10partition_numbers, - log10partition_numbers .+ 2, color = (:black, 0.5), linestyle=:dash, linewidth = lw)
display(fig)
##
#=
log10cumulantsrelativeerror = log10.(abs.(cumulants_list_model ./ reshape(cumulants_list, (1, length(observables))) .- 1))
ms = 20
lw = 5
fig = Figure()
ax = Axis(fig[1, 1])
for i in 1:5
    scatter!(ax, log10partition_numbers, log10cumulantsrelativeerror[:, i], color = (colors[i]), markersize = ms)
end
lines!(ax, log10partition_numbers, - log10partition_numbers .+ 0, color = (:black, 0.5), linestyle=:dash, linewidth = lw)
lines!(ax, log10partition_numbers, - 0.5 .* log10partition_numbers .+ 0, color = (:black, 0.5), linewidth = lw)
display(fig)
=#





