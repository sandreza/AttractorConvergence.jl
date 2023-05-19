using GLMakie, MarkovChainHammer.BayesianMatrix, HDF5, Statistics, ProgressBars
using LinearAlgebra, AttractorConvergence

figure_directory = pwd() * "/figures"
isdir(pwd() * "/figures") ? nothing : mkdir(pwd() * "/figures")

@info "loading data"
hfile = h5open(pwd() * "/data/embedding.hdf5", "r")
markov_chain = read(hfile["markov_chain"])
s_markov_chain = read(hfile["symmetrized markov chain"])
Δt = read(hfile["dt"])
close(hfile)
hfile = h5open(pwd() * "/data/lorenz.hdf5", "r")
m_timeseries = read(hfile["timeseries"])
s_timeseries = read(hfile["symmetrized timeseries"])
close(hfile)
# read centers list from file 
hfile = h5open(pwd() * "/data/kmeans.hdf5", "r")
centers_matrix = read(hfile["centers"])
levels = read(hfile["levels"])
close(hfile)
centers_list = [[centers_matrix[:, 1, i], centers_matrix[:, 2, i]] for i in 1:size(centers_matrix)[3]]
##
@info "computing eigendecomposition"
include("compute_eigenvalue_decomposition.jl") # Perhaps save the data somewhere? Or the first n-eigenvalues / eigenvectors and so forth?
##
@info "creating figures"
figure_number = 1
include("tree_refinement.jl")
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)
println("done with ", figure_number)
figure_number += 1
##
include("statistics_convergence.jl")
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)
println("done with ", figure_number)
figure_number += 1
##
include("eigenvalue_convergence.jl")
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)
println("done with ", figure_number)
figure_number += 1

save(figure_directory * "/Figure" * string(figure_number) * ".png", fig_eig)
println("done with ", figure_number)
figure_number += 1

##
include("koopman_modes.jl")
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)
println("done with ", figure_number)
figure_number += 1

save(figure_directory * "/Figure" * string(figure_number) * ".png", fig2)
println("done with ", figure_number)
figure_number += 1

nothing