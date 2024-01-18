using HDF5, MarkovChainHammer, ProgressBars, LinearAlgebra, Statistics, Random, SparseArrays
using StateSpacePartitions, GLMakie
@info "loading data"

data_directory = "/real_data"
figure_directory = "/unstructured_figures"
hfile = h5open(pwd() * data_directory  * "/eigenvalues.hdf5", "r")
M = 25
q_lambdas = zeros(M)
pf1_lambdas = zeros(M)
pf10_lambdas = zeros(M)
pf100_lambdas = zeros(M)
partitions = zeros(M)
for index in 1:M
    q_lambdas[index] = read(hfile["generator koopman eigenvalue $index"] )
    pf1_lambdas[index] = log(read(hfile["perron_frobenius 1 koopman eigenvalue $index"])) / 1e-4
    pf10_lambdas[index] = log(read(hfile["perron_frobenius 10 koopman eigenvalue $index"])) / 1e-3
    pf100_lambdas[index] = log(read(hfile["perron_frobenius 100 koopman eigenvalue $index"])) /1e-2
    partitions[index] = length(read(hfile["generator koopman eigenvector $index"]))
end
close(hfile)
@info "done loading data now plotting"

ls = 20
fig = Figure()
ax = Axis(fig[1,1]; xlabel = "log10(partition number)", ylabel = "eigenvalue", xticklabelsize=ls, yticklabelsize=ls, xlabelsize=ls, ylabelsize=ls)
x = log10.(partitions)
scatterlines!(ax, x, q_lambdas,     label = "generator", color = (:blue, 0.5))
scatterlines!(ax, x, pf1_lambdas,   label = "perron_frobenius 1", color = (:purple, 0.5))
scatterlines!(ax, x, pf10_lambdas,  label = "perron_frobenius 10", color = (:orange, 0.5))
scatterlines!(ax, x, pf100_lambdas, label = "perron_frobenius 100", color = (:green, 0.5))
#=
scatterlines!(ax, x, log10.(abs.(q_lambdas)),     label = "generator", color = (:blue, 0.5))
scatterlines!(ax, x, log10.(abs.(pf1_lambdas)),   label = "perron_frobenius 1", color = (:purple, 0.5))
scatterlines!(ax, x, log10.(abs.(pf10_lambdas)),  label = "perron_frobenius 10", color = (:orange, 0.5))
scatterlines!(ax, x, log10.(abs.(pf100_lambdas)), label = "perron_frobenius 100", color = (:green, 0.5))
=#
axislegend(ax, position=:rb, framecolor=(:grey, 0.5), patchsize=(10, 10), markersize=10, labelsize=ls)
display(fig)

save(pwd() * figure_directory * "/eigenvalue_convergence_error.png", fig)