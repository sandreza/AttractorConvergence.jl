using ParallelKMeans, HDF5, SparseArrays
using MarkovChainHammer, AttractorConvergence
using ProgressBars, LinearAlgebra, Statistics, Random
using MarkovChainHammer.BayesianMatrix
import MarkovChainHammer.TransitionMatrix: generator, holding_times, steady_state, perron_frobenius, entropy
import MarkovChainHammer.Utils: histogram, autocovariance

# random seed for reproducibility
Random.seed!(12345)
tic = time()
ticci = time()

# create data directory if it's not there
data_directory = "/nobackup1/sandre/AttractorConvergence/data"
isdir(data_directory ) ? nothing : mkdir(data_directory)

##
# generate Lorenz data
if isfile(data_directory  * "/lorenz.hdf5") 
    @info "lorenz data already exists. skipping data generation"
else
    @info "generating Lorenz data"
    include("lorenz.jl")
end
##
include("utils.jl")
##
if isfile(data_directory  * "/embedding.hdf5") 
    @info "embedding data already exists. skipping data generation"
else
    @info "computing embedding"
    include("kmeans_and_embedding.jl")
end
##
if isfile(data_directory  * "/eigenvalues.hdf5") 
    @info "eigenvalue data already exists. skipping data generation"
else
    include("generators_and_eigenvalues.jl")
    include("generators_and_eigenvalues_continue.jl")
end
##
if isfile(data_directory  * "/koopman_timeseries.hdf5") 
    @info "eigenvalue data already exists. skipping data generation"
else
    @info "computing Koopman eigenfunctions"
    include("koopman_timeseries.jl")
end
##
if isfile(data_directory  * "/time_mean_statistics.hdf5") 
    @info "time averaged steady state statistics exists"
else
    @info "computing time averaged steady state statistics"
    include("steady_state_statistics_timeseries.jl")
end


if isfile(data_directory  * "/ensemble_mean_statistics.hdf5")
    @info "ensemble averaged steady state statistics exists"
else
    @info "computing ensemble averaged steady state statistics"
    include("steady_state_statistics_ensemble.jl")

end

if isfile(data_directory  * "/temporal_autocovariance.hdf5")
    @info "temporal autocovariance exists"
else
    @info "computing temporal autocorrelations"
    include("temporal_autocorrelations_timeseries.jl")
end



tacca = time()

println("Time for data generation: ", (tacca - tic)/(60 * 60), " hours")