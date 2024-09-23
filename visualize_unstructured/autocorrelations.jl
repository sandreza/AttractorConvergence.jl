using StateSpacePartitions, MarkovChainHammer

hfile = h5open(data_directory * "temporal_autocovariance.hdf5", "r")
centers_hfile = h5open(data_directory  * "/centers.hdf5", "r")
zautocorrelation = read(hfile["time mean autocovariance"])
generator_autocorrelation = Vector{Float64}[]
perron_frobenius_1_autocorrelation = Vector{Float64}[]
perron_frobenius_10_autocorrelation = Vector{Float64}[]
perron_frobenius_100_autocorrelation = Vector{Float64}[]
partition_number = Int64[]
for i in [12, 16, 20, 24]
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
for i in eachindex(generator_autocorrelation)
    if i == 4
        ax = Axis(fig[i, 1]; xlabel = "time", ylabel = "Cells = $(partition_number[i])")
    elseif i == 1
        ax = Axis(fig[i, 1]; title = "Generator", ylabel = "Cells = $(partition_number[i])")
    else
        ax = Axis(fig[i, 1]; ylabel = "Cells = $(partition_number[i])")
    end
    lines!(ax, truth_ts, zautocorrelation, color=(:blue, op1), linewidth=lw)
    lines!(ax, generator_ts, generator_autocorrelation[i], color=(:red, op2), linewidth=lw)
    xlims!(ax, 0, tmax)
    ylims!(ax, zautomin, zautomax)
    if i ≤ 3
        hidexdecorations!(ax)
    end
end
for i in eachindex(perron_frobenius_1_autocorrelation)
    if i == 4
        ax = Axis(fig[i, 2]; xlabel = "time")
    elseif i == 1
        ax = Axis(fig[i, 2]; title = "Perron-Frobenius (τ = 10⁻³)")
    else
        ax = Axis(fig[i, 2])
    end
    lines!(ax, truth_ts, zautocorrelation, color=(:blue, op1), linewidth=lw)
    lines!(ax, perron_frobenius_1_ts, perron_frobenius_1_autocorrelation[i], color=(:red, op2), linewidth=lw)
    xlims!(ax, 0, tmax)
    ylims!(ax, zautomin, zautomax)
    if i ≤ 3
        hidexdecorations!(ax)
    end
    hideydecorations!(ax)
end
for i in eachindex(perron_frobenius_10_autocorrelation)
    if i == 4
        ax = Axis(fig[i, 3]; xlabel = "time")
    elseif i == 1
        ax = Axis(fig[i, 3]; title = "Perron-Frobenius (τ = 10⁻²)")
    else
        ax = Axis(fig[i, 3])
    end
    lines!(ax, truth_ts, zautocorrelation, color=(:blue, op1), linewidth=lw)
    lines!(ax, perron_frobenius_10_ts, perron_frobenius_10_autocorrelation[i], color=(:red, op2), linewidth=lw)
    xlims!(ax, 0, tmax)
    ylims!(ax, zautomin, zautomax)
    if i ≤ 3
        hidexdecorations!(ax)
    end
    hideydecorations!(ax)
end
for i in eachindex(perron_frobenius_100_autocorrelation)
    if i == 4
        ax = Axis(fig[i, 4]; xlabel = "time")
    elseif i == 1
        ax = Axis(fig[i, 4]; title = "Perron-Frobenius (τ = 10⁻¹)")
    else
        ax = Axis(fig[i, 4])
    end
    lines!(ax, truth_ts, zautocorrelation, color=(:blue, op1), linewidth=lw)
    lines!(ax, perron_frobenius_100_ts, perron_frobenius_100_autocorrelation[i], color=(:red, op2), linewidth=lw)
    xlims!(ax, 0, tmax)
    ylims!(ax, zautomin, zautomax)
    if i ≤ 3
        hidexdecorations!(ax)
    end
    hideydecorations!(ax)
end

# save("unstructured_figures" * "/Figure5.png", fig)
