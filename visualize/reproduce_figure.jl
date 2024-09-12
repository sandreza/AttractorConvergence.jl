using CairoMakie, MarkovChainHammer.BayesianMatrix, HDF5, Statistics, ProgressBars
using LinearAlgebra, AttractorConvergence

tic = time()

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
# decide how many points on scatter to plot, only need Δt = 0.1
Δt_plot = 0.01
indskip = maximum([round(Int, Δt_plot / Δt ), 1])
time_final = 2000
indval = argmin([size(m_timeseries)[2] * Δt, time_final])
last_ind = indval == 1 ? size(m_timeseries)[2] : round(Int, time_final / Δt)
inds = 1:indskip:last_ind
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

save(figure_directory * "/Figure" * string(figure_number) * ".png", fig2)
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
#=
include("koopman_modes.jl")
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)
println("done with ", figure_number)
figure_number += 1

save(figure_directory * "/Figure" * string(figure_number) * ".png", fig2)
println("done with ", figure_number)
figure_number += 1
=#
include("koopman_modes.jl")
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig3)
println("done with ", figure_number)
figure_number += 1

##
include("data_resolution_tradeoff.jl")
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)
println("done with ", figure_number)
figure_number += 1

save(figure_directory * "/Figure" * string(figure_number) * ".png", fig2)
println("done with ", figure_number)
figure_number += 1

##
include("partition_comparison.jl")
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)
println("done with ", figure_number)
figure_number += 1
##
toc = time()
println("total time: ", toc - tic, " seconds")
nothing