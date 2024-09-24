using MarkovChainHammer, ProgressBars, LinearAlgebra
using CairoMakie, HDF5
data_directory = "/nobackup1/sandre/AttractorConvergence/data/"

hfile = h5open(data_directory * "koopman_timeseries.hdf5", "r")
centers_hfile = h5open(data_directory  * "/centers.hdf5", "r")

generator_koopman_timeseries = Vector{Float64}[] 
perron_frobenius_1_koopman_timeseries = Vector{Float64}[]
perron_frobenius_10_koopman_timeseries = Vector{Float64}[]
perron_frobenius_100_koopman_timeseries = Vector{Float64}[]
partition_number = Int64[]
for i in ([12, 16, 20] .+5)
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

hfile = h5open(data_directory * "lorenz.hdf5", "r")
lorenz_timeseries = read(hfile["timeseries"])
close(hfile)

colors = [:red, :purple, :blue]
inds = 1:1:2001
sign_ind = 1600
ls = 8
lw = 1
op = 0.7
ts = (collect(inds) .-1) * 1e-2
fig = Figure(resolution = (1000, 500)) 
ax = Axis(fig[1,1]; xlabel = "time", ylabel = "Koopman Eigenfunction", title = "Generator")
for (i, kts) in enumerate(generator_koopman_timeseries)
    kts = sign(kts[inds][sign_ind]) .* kts
    kts  = kts ./ maximum(abs.(kts[inds]))
    lines!(ax, ts, kts[inds], color = (colors[i], op), linewidth = lw, label = "Cells = $(partition_number[i])")
    ylims!(ax, -1.1, 1.1)
end
axislegend(ax, position=:lt, framecolor=(:grey, 0.5), patchsize=(8,8), labelsize=ls)
ax = Axis(fig[1,2]; xlabel = "time", ylabel = "Koopman Eigenfunction", title = "Perron-Frobenius (τ = 10⁻³)")
for (i, kts) in enumerate(perron_frobenius_1_koopman_timeseries)
    kts = sign(kts[inds][sign_ind]) .* kts
    kts  = kts ./ maximum(abs.(kts[inds]))
    lines!(ax, ts, kts[inds], color = (colors[i], op), linewidth = lw)
    ylims!(ax, -1.1, 1.1)
end
hideydecorations!(ax)
ax = Axis(fig[1, 3]; xlabel = "time", ylabel = "Koopman Eigenfunction", title = "Perron-Frobenius (τ = 10⁻²)")
for (i, kts) in enumerate(perron_frobenius_10_koopman_timeseries)
    kts = sign(kts[inds][sign_ind]) .* kts
    kts  = kts ./ maximum(abs.(kts[inds]))
    lines!(ax, ts, kts[inds], color = (colors[i], op), linewidth = lw)
    ylims!(ax, -1.1, 1.1)
end
hideydecorations!(ax)

titles = ["x", "y", "z"]
for i in 1:3 
    ax = Axis(fig[2, i]; ylabel = titles[i], title = "Lorenz", xlabel = "time")
    lines!(ax, ts, lorenz_timeseries[i, 1:10:end][inds], color = :black)
end

#=
ax = Axis(fig[2,2])
for (i, kts) in enumerate(perron_frobenius_100_koopman_timeseries)
    if i ≤ 2
    kts = sign(kts[inds][sign_ind]) .* kts
    kts  = kts ./ maximum(abs.(kts[inds]))
    lines!(ax, ts, kts[inds], color = (colors[i], op), linewidth = lw)
    ylims!(ax, -1.1, 1.1)
    end
end
=#



# save("unstructured_figures" * "/Figure6.png", fig)
