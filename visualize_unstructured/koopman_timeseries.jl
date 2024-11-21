using MarkovChainHammer, ProgressBars, LinearAlgebra
using CairoMakie, HDF5

data_directory = "./data/"

hfile = h5open(data_directory * "koopman_timeseries.hdf5", "r")
centers_hfile = h5open(data_directory  * "/centers.hdf5", "r")

generator_koopman_timeseries            = Vector{Float64}[] 
perron_frobenius_1_koopman_timeseries   = Vector{Float64}[]
perron_frobenius_10_koopman_timeseries  = Vector{Float64}[]
perron_frobenius_100_koopman_timeseries = Vector{Float64}[]
partition_number = Int64[]

for i in ([12, 16, 20] .+ 4)
    gkt   = read(hfile["generator koopman timeseries $i"])
    pf1   = read(hfile["perron_frobenius 1 koopman timeseries $i"])
    pf10  = read(hfile["perron_frobenius 10 koopman timeseries $i"])
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
ax = Axis(fig[1,1]; xlabel = L"\text{time}", ylabel = L"\text{Koopman Eigenfunction}", title = L"\text{Generator}")
for (i, kts) in enumerate(generator_koopman_timeseries)
    kts = sign(kts[inds][sign_ind]) .* kts
    kts  = kts ./ maximum(abs.(kts[inds]))
    lines!(ax, ts, kts[inds], color = (colors[i], op), linewidth = lw, label = L"\text{Cells = }%$(partition_number[i])")
    ylims!(ax, -1.1, 1.1)
end
axislegend(ax, position=:lt, framecolor=(:grey, 0.5), patchsize=(8,8), labelsize=ls)
ax = Axis(fig[1,2]; xlabel = L"\text{time}", ylabel = L"\text{Koopman Eigenfunction}", title = "\text{Perron-Frobenius }(\tau = 10^{-3})")
for (i, kts) in enumerate(perron_frobenius_1_koopman_timeseries)
    kts = sign(kts[inds][sign_ind]) .* kts
    kts  = kts ./ maximum(abs.(kts[inds]))
    lines!(ax, ts, kts[inds], color = (colors[i], op), linewidth = lw)
    ylims!(ax, -1.1, 1.1)
end
hideydecorations!(ax)
ax = Axis(fig[1, 3]; xlabel = L"\text{time}", ylabel = L"\text{Koopman Eigenfunction}", title = "\text{Perron-Frobenius }(\tau = 10^{-2})")
for (i, kts) in enumerate(perron_frobenius_10_koopman_timeseries)
    kts = sign(kts[inds][sign_ind]) .* kts
    kts  = kts ./ maximum(abs.(kts[inds]))
    lines!(ax, ts, kts[inds], color = (colors[i], op), linewidth = lw)
    ylims!(ax, -1.1, 1.1)
end
hideydecorations!(ax)

titles = ["x", "y", "z"]
for i in 1:3 
    ax = Axis(fig[2, i]; ylabel = L"%$(titles[i]) trajectory", xlabel = L"\text{time}")
    lines!(ax, ts, lorenz_timeseries[i, 1:10:end][inds], color = :black)
end

figure_directory = pwd() * "/unstructured_figures"; figure_number = 8; 

save(figure_directory * "/Figure" * string(figure_number) * ".eps", fig)
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)
