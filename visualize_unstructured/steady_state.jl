@info "opening centers"
centerslist = []
hfile = h5open(pwd() * data_directory  * "/centers.hdf5", "r")
for i in ProgressBar(eachindex(probabilities))
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

observables_list = zeros(length(observables))
for j in ProgressBar(eachindex(observables))
    N = length(eachcol(joined_timeseries))
    for state in ProgressBar(eachcol(joined_timeseries))
        observables_list[j] += observables[j](state) / N
    end
end

##
observables_list_model = zeros(10, length(observables))
for i in 1:10
    p = MarkovChainHammer.Utils.steady_state(Ps[i])
    for j in eachindex(p)
        for k in eachindex(observables)
            observables_list_model[i, k] += observables[k](centerslist[i][:, j]) .* p[j]
        end
    end
end
##

errors = log10.(abs.(observables_list_model .- reshape(observables_list, (1, length(observables)))))

using UnicodePlots
partition_numbers = [maximum(coarse_markov_chain[:, i]) for i in 1:10]
scatterplot(log10.(partition_numbers), errors[:, 1])

