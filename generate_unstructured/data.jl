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
data_directory = "/test_data"
isdir(pwd() * data_directory ) ? nothing : mkdir(pwd() * data_directory )

##
# generate Lorenz data
if isfile(pwd() * data_directory  * "/lorenz.hdf5") #unideal because just checking for one
    @info "lorenz data already exists. skipping data generation"
else
    include("lorenz.jl")
end
##
include("utils.jl")
##
# create embedding 
if isfile(pwd() * data_directory  * "/embedding.hdf5") #unideal because just checking for one
    @info "embedding data already exists. skipping data generation"
else
    include("kmeans.jl")
end
##
# 10^(1/3 * (3 + 15) ) seems like a good number of bins (0, 1, ..., 15) for plotting