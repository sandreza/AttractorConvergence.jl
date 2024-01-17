using MarkovChainHammer, HDF5, ProgressBars
using GLMakie

import MarkovChainHammer.Utils: histogram
eigenvalues_hfile = h5open(pwd() * data_directory  * "/eigenvalues.hdf5", "r")
centers_hfile = h5open(pwd() * data_directory  * "/centers.hdf5", "r")
ps = Vector{Float64}[]
centers = Vector{Float64}[]
for i in 1:25
    p = read(eigenvalues_hfile["generator steady state $i"] )
    zcenter = read(centers_hfile["centers $i"])[3, :]
    push!(ps, p)
    push!(centers, zcenter)
end
close(eigenvalues_hfile)
close(centers_hfile)

dt_skip = 0.01
@info "loading data"
hfile = h5open(pwd() * data_directory  * "/lorenz.hdf5", "r")
dt = read(hfile["dt"])
skip = round(Int, dt_skip/dt)
joined_timeseries = read(hfile["timeseries"])[:, 1:skip:end]
close(hfile)
@info "done loading timeseries"

##
values_e = Vector{Float64}[]
probabilities_e = Vector{Float64}[]
values_t = Vector{Float64}[]
probabilities_t = Vector{Float64}[]
for i in ProgressBar(1:25)
    iii = i + 0
    bin_numbers = maximum([length(ps[iii]) ÷ 1000, 20])
    values, probabilities = histogram(centers[iii]; bins=bin_numbers, normalization=ps[iii], custom_range = (0, 50))
    values2, probabilities2 = histogram(joined_timeseries[3, :]; bins=bin_numbers, custom_range = (0, 50))
    push!(values_e, values)
    push!(probabilities_e, probabilities)
    push!(values_t, values2)
    push!(probabilities_t, probabilities2)
end

##
fig = Figure() 
labelsize = 40
options = (; xlabel="z", ylabel="Probability", titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
for i in 1:25
    ii = (i-1) ÷ 5 + 1
    jj = (i-1) % 5 + 1
    ax = Axis(fig[ii, jj])
    iii = i + 0
    bin_numbers = maximum([length(ps[iii]) ÷ 1000, 20])
    values = values_e[iii]
    probabilities = probabilities_e[iii]
    values2 = values_t[iii]
    probabilities2 = probabilities_t[iii]
    barplot!(ax, values, probabilities; color=(:red, 0.25), gap=0.0, options...)
    barplot!(ax, values2, probabilities2; color=(:blue, 0.25), gap=0.0, options...)
end
display(fig)

save(pwd() * figure_directory * "/histogram_convergence.png", fig)