using ParallelKMeans, HDF5
using GLMakie, Revise
using MarkovChainHammer, AttractorConvergence
using ProgressBars, LinearAlgebra, Statistics, Random
using MarkovChainHammer.BayesianMatrix
import MarkovChainHammer.TransitionMatrix: generator, holding_times, steady_state, perron_frobenius, entropy
import MarkovChainHammer.Utils: histogram, autocovariance

# random seed for reproducibility
Random.seed!(12345)
tic = time()

# create data directory if it's not there
isdir(pwd() * "/data") ? nothing : mkdir(pwd() * "/data")

##
# generate Lorenz data
if isfile(pwd() * "/data/lorenz.hdf5") #unideal because just checking for one
    @info "lorenz data already exists. skipping data generation"
else
    include("lorenz.jl")
end
##
# generate kmeans and save
if isfile(pwd() * "/data/kmeans.hdf5") #unideal because just checking for one
    @info "kmeans data already exists. skipping data generation"
else
    include("kmeans.jl")
end
##
# create embedding 
if isfile(pwd() * "/data/embedding.hdf5") #unideal because just checking for one
    @info "embedding data already exists. skipping data generation"
else
    include("embedding.jl")
end
##
# create structured embedding 
if isfile(pwd() * "/data/structured_embedding.hdf5") #unideal because just checking for one
    @info "structured embedding data already exists. skipping data generation"
else
    include("structured_embedding.jl")
end

##
toc = time()
println("The total amount of time to reproduce data is $(toc-tic) seconds.")