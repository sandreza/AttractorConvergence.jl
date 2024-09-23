using MarkovChainHammer, ProgressBars, LinearAlgebra
using CairoMakie, HDF5
data_directory = "/nobackup1/sandre/AttractorConvergence/old_data/"

hfile = h5open(data_directory * "koopman_timeseries.hdf5", "r")
centers_hfile = h5open(data_directory  * "/centers.hdf5", "r")

generator_koopman_timeseries = Vector{Float64}[] 
perron_frobenius_1_koopman_timeseries = Vector{Float64}[]
perron_frobenius_10_koopman_timeseries = Vector{Float64}[]
perron_frobenius_100_koopman_timeseries = Vector{Float64}[]
partition_number = Int64[]
for i in [16, 18, 20]
    gkt = read(hfile["generator koopman timeseries $i"])
    pf1 = read(hfile["perron_frobenius 1 koopman timeseries $i"])
    pf10 = read(hfile["perron_frobenius 10 koopman timeseries $i"])
    pf100 = read(hfile["perron_frobenius 100 koopman timeseries $i"])

    push!(generator_koopman_timeseries, gkt)
    push!(perron_frobenius_1_koopman_timeseries, pf1)
    push!(perron_frobenius_10_koopman_timeseries, pf10)
    push!(perron_frobenius_100_koopman_timeseries, pf100)

    push!(partition_number, size(read(centers_hfile["centers $i"]), 2))
end

close(hfile)
close(centers_hfile)

colors = [:red, :purple, :blue]
inds = 1:1:2001
sign_ind = 1600
ls = 8
lw = 1
ts = (collect(inds) .-1) * 1e-2
fig = Figure() 
ax = Axis(fig[1,1])
for (i, kts) in enumerate(generator_koopman_timeseries)
    kts = sign(kts[inds][sign_ind]) .* kts
    kts  = kts ./ maximum(abs.(kts[inds]))
    lines!(ax, ts, kts[inds], color = colors[i], linewidth = lw, label = "Cells = $(partition_number[i])")
    ylims!(ax, -1.1, 1.1)
end
axislegend(ax, position=:lt, framecolor=(:grey, 0.5), patchsize=(8,8), labelsize=ls)
ax = Axis(fig[1,2])
for (i, kts) in enumerate(perron_frobenius_1_koopman_timeseries)
    kts = sign(kts[inds][sign_ind]) .* kts
    kts  = kts ./ maximum(abs.(kts[inds]))
    lines!(ax, ts, kts[inds], color = colors[i], linewidth = lw)
    ylims!(ax, -1.1, 1.1)
end
ax = Axis(fig[2,1])
for (i, kts) in enumerate(perron_frobenius_10_koopman_timeseries)
    kts = sign(kts[inds][sign_ind]) .* kts
    kts  = kts ./ maximum(abs.(kts[inds]))
    lines!(ax, ts, kts[inds], color = colors[i], linewidth = lw)
    ylims!(ax, -1.1, 1.1)
end
ax = Axis(fig[2,2])
for (i, kts) in enumerate(perron_frobenius_100_koopman_timeseries)
    kts = sign(kts[inds][sign_ind]) .* kts
    kts  = kts ./ maximum(abs.(kts[inds]))
    lines!(ax, ts, kts[inds], color = colors[i], linewidth = lw)
    ylims!(ax, -1.1, 1.1)
end


save("unstructured_figures" * "/Figure6.png", fig)
