using StateSpacePartitions, MarkovChainHammer
using HDF5, CairoMakie

data_directory = "data/"
hfile = h5open(data_directory * "temporal_autocovariance.hdf5", "r")
centers_hfile = h5open(data_directory  * "/centers.hdf5", "r")
zautocorrelation = read(hfile["time mean autocovariance"])
generator_autocorrelation = Vector{Float64}[]
perron_frobenius_1_autocorrelation = Vector{Float64}[]
perron_frobenius_10_autocorrelation = Vector{Float64}[]
perron_frobenius_100_autocorrelation = Vector{Float64}[]
partition_number = Int64[]
for i in ([12, 16, 20, 24] .+ 1)
    g = read(hfile["ensemble mean autocovariance generator $i"])
    pf1 = read(hfile["ensemble mean autocovariance perron_frobenius 1 $i"])
    pf10 = read(hfile["ensemble mean autocovariance perron_frobenius 10 $i"])
    pf100 = read(hfile["ensemble mean autocovariance perron_frobenius 100 $i"])
    push!(generator_autocorrelation, g)
    push!(perron_frobenius_1_autocorrelation, pf1)
    push!(perron_frobenius_10_autocorrelation, pf10)
    push!(perron_frobenius_100_autocorrelation, pf100)
    push!(partition_number, size(read(centers_hfile["centers $i"]), 2))
end
close(hfile)
close(centers_hfile)

##

op1 = 0.5
op2 = 0.5
lw = 3
tmax = 20
zautomin = -50
zautomax = 75
fig  = Figure(resolution = (1000, 1000))
truth_ts = range(0, 40, length= length(zautocorrelation) + 1)[1:end-1]
generator_ts = range(0, 40, length= length(generator_autocorrelation[1]))
perron_frobenius_1_ts = range(0, 40, length= length(perron_frobenius_1_autocorrelation[1]))
perron_frobenius_10_ts = range(0, 40, length= length(perron_frobenius_10_autocorrelation[1]))
perron_frobenius_100_ts = range(0, 40, length= length(perron_frobenius_100_autocorrelation[1]))

xticksvisible   = ([0, 5, 10, 15, 20], [L"0", L"5", L"10", L"15", L"20"])
xticksinvisible = ([0, 5, 10, 15, 20], ["", "", "", "", ""])
yticksvisible   = ([-50, -25, 0, 25, 50, 75], [L"-50", L"-25", L"0", L"25", L"50", L"75"])
yticksinvisible = ([-50, -25, 0, 25, 50, 75], ["", "", "", "", "", ""])

for i in eachindex(generator_autocorrelation)
    if i == 4
        ax = Axis(fig[i, 1]; 
                  xlabel = L"\text{time}", 
                  ylabel = L"\text{Cells = } %$(partition_number[i])",
                  xticks = xticksvisible,
                  yticks = yticksvisible)
    elseif i == 1
        ax = Axis(fig[i, 1]; title = L"\text{Generator}", 
                  ylabel = L"\text{Cells = } %$(partition_number[i])",
                  xticks = xticksinvisible,
                  yticks = yticksvisible)
    else
        ax = Axis(fig[i, 1]; 
                  ylabel = L"\text{Cells = } %$(partition_number[i])",
                  xticks = xticksinvisible,
                  yticks = yticksvisible)
    end
    lines!(ax, truth_ts, zautocorrelation, color=(:blue, op1), linewidth=lw)
    lines!(ax, generator_ts, generator_autocorrelation[i], color=(:red, op2), linewidth=lw)
    xlims!(ax, 0, tmax)
    ylims!(ax, zautomin, zautomax)
end
for i in eachindex(perron_frobenius_1_autocorrelation)
    if i == 4
        ax = Axis(fig[i, 2]; xlabel = L"\text{time}",
                  xticks = xticksvisible,
                  yticks = yticksinvisible)
    elseif i == 1
        ax = Axis(fig[i, 2]; title = L"\text{Perron-Frobenius }(\tau = 10^{-3})",
                  xticks = xticksinvisible,
                  yticks = yticksinvisible)
    else
        ax = Axis(fig[i, 2],
                  xticks = xticksinvisible,
                  yticks = yticksinvisible)
    end
    lines!(ax, truth_ts, zautocorrelation, color=(:blue, op1), linewidth=lw)
    lines!(ax, perron_frobenius_1_ts, perron_frobenius_1_autocorrelation[i], color=(:red, op2), linewidth=lw)
    xlims!(ax, 0, tmax)
    ylims!(ax, zautomin, zautomax)
end
for i in eachindex(perron_frobenius_10_autocorrelation)
    if i == 4
        ax = Axis(fig[i, 3]; xlabel = L"\text{time}",
                  xticks = xticksvisible,
                  yticks = yticksinvisible)
    elseif i == 1
        ax = Axis(fig[i, 3]; title = L"\text{Perron-Frobenius }(\tau = 10^{-3})",
                  xticks = xticksinvisible,
                  yticks = yticksinvisible)
    else
        ax = Axis(fig[i, 3],
                  xticks = xticksinvisible,
                  yticks = yticksinvisible)
    end
    lines!(ax, truth_ts, zautocorrelation, color=(:blue, op1), linewidth=lw)
    lines!(ax, perron_frobenius_10_ts, perron_frobenius_10_autocorrelation[i], color=(:red, op2), linewidth=lw)
    xlims!(ax, 0, tmax)
    ylims!(ax, zautomin, zautomax)
end
for i in eachindex(perron_frobenius_100_autocorrelation)
    if i == 4
        ax = Axis(fig[i, 4]; xlabel = L"\text{time}",
                  xticks = xticksvisible,
                  yticks = yticksinvisible)
    elseif i == 1
        ax = Axis(fig[i, 4]; title = L"\text{Perron-Frobenius }(\tau = 10^{-1})",
                  xticks = xticksinvisible,
                  yticks = yticksinvisible)
    else
        ax = Axis(fig[i, 4],
                  xticks = xticksinvisible,
                  yticks = yticksinvisible)
    end
    lines!(ax, truth_ts, zautocorrelation, color=(:blue, op1), linewidth=lw)
    lines!(ax, perron_frobenius_100_ts, perron_frobenius_100_autocorrelation[i], color=(:red, op2), linewidth=lw)
    xlims!(ax, 0, tmax)
    ylims!(ax, zautomin, zautomax)
end

figure_directory = pwd() * "/unstructured_figures"; figure_number = 6; 

save(figure_directory * "/Figure" * string(figure_number) * ".eps", fig)
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)