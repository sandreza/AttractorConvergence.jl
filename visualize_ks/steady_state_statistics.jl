using HDF5, GLMakie, StatsBase, MarkovChainHammer

probability_path = data_directory * "/probability.hdf5"
if isfile(probability_path)
    @info "probability.hdf5 exists; skipping probability generation."
else
    hfile = h5open(data_directory * "/embedding.hdf5", "r")
    probability_hfile = h5open(probability_path, "w")
    for j in ProgressBar(1:24)
        markov_chains = read(hfile["coarse_markov_chains $j"])
        nmax = maximum(markov_chains)
        count_vector = StatsBase.counts(markov_chains[:], 1:nmax)
        p = count_vector ./ length(markov_chains)
        probability_hfile["probability $j"] =  p
    end
    markov_chains = read(hfile["markov_chain"])
    nmax = maximum(markov_chains)
    count_vector = StatsBase.counts(markov_chains[:], 1:nmax)
    p = count_vector ./ length(markov_chains)
    probability_hfile["probability"] = p
    close(hfile)
    close(probability_hfile)
end


if isfile(data_directory * "/observables.hdf5")
    @info "observables.hdf5 exists; skipping observables generation."
else
    gfiles = h5open(data_directory * "/observables.hdf5", "w")
    observables_t = [x -> mean(x .^ 2), x -> mean(x .^ 4), x -> x[1] * x[3], x -> x[1] * x[5]]
    hfile = h5open(data_directory * "/ks.hdf5", "r")
    timeseries = read(hfile["timeseries"])
    close(hfile)
    for (j, observable) in enumerate(observables_t)
        gt = mean([observable(timeseries[:, i]) for i in ProgressBar(1:size(timeseries)[2])])
        gfiles["obs $j gt"] = gt
    end
    close(gfiles)

    gfiles = h5open(data_directory * "/observables.hdf5", "r+")
    observables = [x -> mean(x .^ 2), x -> mean(x .^ 4), x -> x[1] * x[2], x -> x[1] * x[3]]
    @info "loading data for kmeans"

    hfile = h5open(data_directory * "/centers.hdf5", "r")
    pfile = h5open(data_directory * "/probability.hdf5", "r")
    for j in ProgressBar(1:24)
        centers = read(hfile["centers $j"])
        p = read(pfile["probability $j"])
        for (k, g) in enumerate(observables)
            gfiles["obs $k coarse $j"] = sum([g(centers[:, i]) for i in ProgressBar(1:size(centers)[2])] .* p)
        end
    end
    centers = read(hfile["centers"])
    p = read(pfile["probability"])
    for (k, g) in enumerate(observables)
        gfiles["obs $k"] = sum([g(centers[:, i]) for i in ProgressBar(1:size(centers)[2])] .* p)
    end
    close(hfile)
    close(pfile)
    close(gfiles)
end

center_file = h5open(data_directory * "/centers.hdf5", "r")
cell_numbers = [size(read(center_file["centers $j"]), 2) for j in 1:24]
cell_numbers = vcat(cell_numbers..., size(read(center_file["centers"]), 2))
close(center_file)

hfile = h5open(data_directory * "/observables.hdf5", "r")
observables_gt = [read(hfile["obs $k gt"]) for k in 1:4]
obs_vectors = []
delta_vectors = []
for k in 1:4
    obs_vector = [read(hfile["obs $k coarse $j"]) for j in 1:24]
    obs_vector = vcat(obs_vector..., read(hfile["obs $k"]))
    delta_vector = abs.(obs_vector .- observables_gt[k]) / abs(observables_gt[k])
    push!(obs_vectors, obs_vector)
    push!(delta_vectors, delta_vector)
end
close(hfile)


fig = Figure(resolution = (750, 400), fontsize = 18)
ax = Axis(fig[1, 1];
          xlabel = L"\text{Cells}",
          ylabel = L"\text{Relative error}",
          yticks = ([-2, -1.5, -1.0, -0.5, 0], [L"10^{-2}", L"10^{-1.5}", L"10^{-1.0}", L"10^{-0.5}", L"10^{0}"]),
          xticks = ([2, 4, 6], [L"10^2", L"10^4", L"10^6"]))
colors = [:red, :green, :blue, :orange]
for k in 1:4
    yvals = log10.(abs.(obs_vectors[k] .- observables_gt[k]) ./ abs(observables_gt[k]))
    label_k = k == 1 ? L"\frac{1}{L}\int_0^L u^2(x) dx" :
              k == 2 ? L"\frac{1}{L}\int_0^L u^4(x) dx" :
              k == 3 ? L"u(0) u(1.6)" :
                       L"u(0) u(2.7)"
    scatterlines!(ax, log10.(cell_numbers), yvals; markersize = 10, marker = (k == 1 ? '●' : k == 2 ? '◆' : k == 3 ? '■' : :hexagon),
                  linewidth = 0.3, color = colors[k], label = label_k)
end
ylims!(ax, -2.1, 0.1)
lines!(ax, log10.(cell_numbers), 0.02 .- log10.(cell_numbers)/4, color = (:black, 0.5), linestyle = :dash, linewidth = 1.5, label = L"\text{slope }-1/4")
axislegend(ax, position = :lb, framecolor = (:grey, 0.5), framevisible = false)
display(fig)

figure_directory = "KSFigures";
save(figure_directory * "/steady_state_statistics.png", fig)

#=
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "log10(# cells)", ylabel = "Relative Error", xlabelsize = 20, ylabelsize = 20, xticklabelsize = 20, yticklabelsize = 20)
colors = [:red, :green, :blue, :orange]
labels = ["mean(x²)", "mean(x⁴)", "x¹x³", "x¹x⁵"]
for k in 1:4
    lines!(ax, log10.(cell_numbers), log10.(abs.(obs_vectors[k] .- observables_gt[k])), color = colors[k], label = labels[k])
    scatter!(ax, log10.(cell_numbers), log10.(abs.(obs_vectors[k] .- observables_gt[k])), color = colors[k])
end
lines!(ax, log10.(cell_numbers), 0.5 .- log10.(cell_numbers)/4, color = (:black, 0.5), linestyle=:dash, linewidth = 2.5, label = 
"slope -1/4")
axislegend(ax, position = :lb)
display(fig)
figure_directory = "KSFigures";
save(figure_directory * "/steady_state_statistics.png", fig)
=#
