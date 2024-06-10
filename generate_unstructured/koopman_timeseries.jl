using HDF5, MarkovChainHammer, ProgressBars, LinearAlgebra, Statistics, Random, SparseArrays
using StateSpacePartitions
first_index = 1

hfile = h5open(data_directory  * "/lorenz.hdf5", "r")
dt = read(hfile["dt"])
close(hfile)

hfile = h5open(data_directory  * "/embedding.hdf5", "r")
coarse_markov_chain = read(hfile["coarse_markov_chains $first_index"])
probability = read(hfile["probability"])
coarse_probabilities = read(hfile["coarse_probabilities"])
close(hfile)

hfile = h5open(data_directory  * "/centers.hdf5", "r")
centers = read(hfile["centers $first_index"])
close(hfile)

dt_min = 1e-2
@info "computing koopman mode timeseries"
kthfile = h5open(data_directory  * "/koopman_timeseries.hdf5", "w")
for (index, probability) in ProgressBar(enumerate(coarse_probabilities))
    nhfile = h5open(data_directory  * "/embedding.hdf5", "r")
    coarse_markov_chain = read(nhfile["coarse_markov_chains $index"])
    close(nhfile)
    hfile = h5open(data_directory  * "/eigenvalues.hdf5", "r")
    w = read(hfile["generator koopman eigenvector $index"] )
    close(hfile)
    skip = maximum([round(Int, dt_min/dt ), 1])
    skipdt = skip * dt 
    markov_chain = coarse_markov_chain[1:skip:end]
    koopman_timeseries = zeros(length(markov_chain))
    for i in eachindex(koopman_timeseries)
        koopman_timeseries[i] = w[markov_chain[i]]
    end
    kthfile["generator koopman timeseries $index"] = koopman_timeseries 
    kthfile["generator timeseries dt $index"] = skipdt
    kthfile["generator timeseries skip $index"] = skip
    for k in [1, 10, 100]
        hfile = h5open(data_directory  * "/eigenvalues.hdf5", "r")
        w = read(hfile["perron_frobenius $k koopman eigenvector $index"] )
        kdt = read(hfile["perron_frobenius $k dt $index"])
        close(hfile)
        skip = maximum([round(Int, dt_min/dt ), 1])
        skipdt = skip * dt 
        markov_chain = coarse_markov_chain[1:skip:end]
        koopman_timeseries = zeros(length(markov_chain))
        for i in eachindex(koopman_timeseries)
            koopman_timeseries[i] = w[markov_chain[i]]
        end
        kthfile["perron_frobenius $k koopman timeseries $index"] = koopman_timeseries
        kthfile["perron_frobenius $k timeseries dt $index"] = skipdt
        kthfile["perron_frobenius $k timeseries skip $index"] = skip
    end
end
close(kthfile)
