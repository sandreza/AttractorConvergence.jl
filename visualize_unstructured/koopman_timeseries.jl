using MarkovChainHammer, ProgressBars, LinearAlgebra
using CairoMakie, HDF5

CairoMakie.activate!()

data_directory = "./data/"
data_directory = "/nobackup1/sandre/AttractorConvergence/data/"

hfile = h5open(data_directory * "koopman_timeseries.hdf5", "r")
centers_hfile = h5open(data_directory  * "/centers.hdf5", "r")

generator_koopman_timeseries            = Vector{Float64}[] 
perron_frobenius_1_koopman_timeseries   = Vector{Float64}[]
perron_frobenius_10_koopman_timeseries  = Vector{Float64}[]
perron_frobenius_100_koopman_timeseries = Vector{Float64}[]
partition_number = Int64[]

for i in ([12, 16, 21] .+ 4)
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

xvisible     = ([0, 5, 10, 15, 20],    [L"0", L"5", L"10", L"15", L"20"])
xinvisible   = ([0, 5, 10, 15, 20],    ["", "", "", "", ""])
yPFvisible   = ([-1, -0.5, 0, 0.5, 1], [L"-1", L"-0.5", L"0", L"0.5", L"1"])

ytrajvisible = [([-20, -10, 0, 10, 20], [L"-20", L"-10", L"0", L"10", L"20"]),
                ([-20, -10, 0, 10, 20], [L"-20", L"-10", L"0", L"10", L"20"]),
                ([10, 20, 30, 40],      [L"10", L"20", L"30", L"40"])]

colors = [:red, :purple, :blue]
inds = 1:1:2001
sign_ind = 1600
ls = 8
lw = 1
op = 0.7
ts = (collect(inds) .-1) * 1e-2
fig = Figure(size = (1000, 500)) 
ax = Axis(fig[1,1]; 
          xlabel = "", 
          ylabel = L"\text{Koopman Eigenfunction}", 
          title  = L"\text{Generator}",
          xticks = xinvisible,
          yticks = yPFvisible)
for (i, kts) in enumerate(generator_koopman_timeseries)
    kts = sign(kts[inds][sign_ind]) .* kts
    kts  = kts ./ maximum(abs.(kts[inds]))
    lines!(ax, ts, kts[inds], color = (colors[i], op), linewidth = lw, label = L"%$(partition_number[i])")
    scatter!(ax, ts[501], kts[501], color = :green, markersize = 10)
    ylims!(ax, -1.1, 1.1)
end
# axislegend(ax, position=:ct, framevisible=false)

Legend(fig[1, 4], ax, L"\text{Cells}") # , lines, [L"%$(partition_number[i])" for i in 1:3])

ax = Axis(fig[1,2]; 
          xlabel = "", 
          ylabel = L"\text{Koopman Eigenfunction}", 
          title  = L"\text{Perron-Frobenius }(\tau = 10^{-3})",
          xticks = xinvisible,
          yticks = yPFvisible)
for (i, kts) in enumerate(perron_frobenius_1_koopman_timeseries)
    kts = sign(kts[inds][sign_ind]) .* kts
    kts  = kts ./ maximum(abs.(kts[inds]))
    lines!(ax, ts, kts[inds], color = (colors[i], op), linewidth = lw)
    scatter!(ax, ts[501], kts[501], color = :green, markersize = 10)
    ylims!(ax, -1.1, 1.1)
end
ax = Axis(fig[1, 3]; 
          xlabel = "", 
          ylabel = L"\text{Koopman Eigenfunction}", 
          title  = L"\text{Perron-Frobenius }(\tau = 10^{-2})",
          xticks = xinvisible,
          yticks = yPFvisible)

lines = []
for (i, kts) in enumerate(perron_frobenius_10_koopman_timeseries)
    kts = sign(kts[inds][sign_ind]) .* kts
    kts  = kts ./ maximum(abs.(kts[inds]))
    push!(lines, lines!(ax, ts, kts[inds], color = (colors[i], op), linewidth = lw))
    scatter!(ax, ts[501], kts[501], color = :green, markersize = 10)
    ylims!(ax, -1.1, 1.1)
end
    
titles = ["x", "y", "z"]
for i in 1:3 
    ax = Axis(fig[2, i]; 
              ylabel = L"%$(titles[i]) - \text{trajectory}", 
              xlabel = L"\text{time}",
              xticks = xvisible,
              yticks = ytrajvisible[i])
    lines!(ax, ts, lorenz_timeseries[i, 1:10:end][inds], color = :black)
    scatter!(ax, ts[501], lorenz_timeseries[i, 1:10:end][501], color = :green, markersize = 10)
end

figure_directory = pwd() * "/unstructured_figures"; figure_number = 8; 

save(figure_directory * "/Figure" * string(figure_number) * ".eps", fig)
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)
