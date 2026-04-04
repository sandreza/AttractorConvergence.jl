using HDF5, CairoMakie
using StateSpacePartitions, Graphs, GraphMakie
using MarkovChainHammer, ProgressBars, LinearAlgebra
using SparseArrays, NetworkLayout, Printf, Random

data_directory = "./data"
@info "loading data for kmeans"
hfile = h5open(data_directory * "/eigenvalues.hdf5", "r")
read(hfile["generator koopman eigenvalue 16"])
read(hfile["generator koopman eigenvalue 20"])
read(hfile["generator koopman eigenvalue 25"])
close(hfile)

cfile = h5open(data_directory * "/centers.hdf5", "r")
close(cfile)
