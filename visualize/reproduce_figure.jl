using GLMakie, MarkovChainHammer.BayesianMatrix, HDF5

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

@info "creating figures"
figure_number = 1
include("tree_refinement.jl")
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)
println("done with ", figure_number)

figure_number += 1
nothing