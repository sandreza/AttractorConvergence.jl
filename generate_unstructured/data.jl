using ParallelKMeans, HDF5, SparseArrays
using MarkovChainHammer, AttractorConvergence
using ProgressBars, LinearAlgebra, Statistics, Random
using MarkovChainHammer.BayesianMatrix
import MarkovChainHammer.TransitionMatrix: generator, holding_times, steady_state, perron_frobenius, entropy
import MarkovChainHammer.Utils: histogram, autocovariance

# random seed for reproducibility
Random.seed!(12345)
tic = time()

# create data directory if it's not there
data_directory = "/storage4/andre/attractor_convergence" * "/real_data"
isdir(data_directory ) ? nothing : mkdir(data_directory)

##
# generate Lorenz data
if isfile(data_directory  * "/lorenz.hdf5") #unideal because just checking for one
    @info "lorenz data already exists. skipping data generation"
else
    include("lorenz.jl")
end
##
include("utils.jl")
##
if isfile(data_directory  * "/embedding.hdf5") #unideal because just checking for one
    @info "embedding data already exists. skipping data generation"
else
    include("kmeans_and_embedding.jl")
end
##
if isfile(data_directory  * "/eigenvalues.hdf5") #unideal because just checking for one
    @info "eigenvalue data already exists. skipping data generation"
else
    include("generators_and_eigenvalues.jl")
end
##
if isfile(data_directory  * "/koopman_timeseries.hdf5") #unideal because just checking for one
    @info "eigenvalue data already exists. skipping data generation"
else
    include("koopman_timeseries.jl")
end

##
# TO DO: autocorrelation data, steady state data