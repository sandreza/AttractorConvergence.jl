using ParallelKMeans, HDF5
using GLMakie, Revise
using MarkovChainHammer, AttractorConvergence
using ProgressBars, LinearAlgebra, Statistics, Random
using MarkovChainHammer.BayesianMatrix
import MarkovChainHammer.TransitionMatrix: generator, holding_times, steady_state, perron_frobenius, entropy
import MarkovChainHammer.Utils: histogram, autocovariance

# random seed for reproducibility
Random.seed!(12345)

# create data directory if it's not there
isdir(pwd() * "/data") ? nothing : mkdir(pwd() * "/data")

# generate Lorenz data
if isfile(pwd() * "/data/lorenz.hdf5") #unideal because just checking for one
    @info "lorenz data already exists. skipping data generation"
else
    include("lorenz.jl")
end

# generate kmeans and save
if isfile(pwd() * "/data/kmeans.hdf5") #unideal because just checking for one
    @info "lorenz data already exists. skipping data generation"
else
    include("kmeans.jl")
end