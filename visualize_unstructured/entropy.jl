using MarkovChainHammer


hfile = h5open(data_directory * "eigenvalues.hdf5", "r")

ps = Vector{Float64}[]
for i in 1:25
    push!(ps, read(hfile["generator steady state $i"]))
end

close(hfile)

entropies = scaled_entropy.(ps)
cells = length.(ps)

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "log10(cells)", ylabel = "log10(scaled entropy)", title = "Scaled Entropy vs. Cells")
scatter!(ax, log10.(cells), log10.(entropies), markersize = 10, color = :black, label = "data")
hlines!(ax, [log10(1)], color = :red, linestyle = :dash)
ylims!(ax, -0.011, 0.001)
xlims!(ax, 0.5, 6.1)

# save(figure_directory * "/Figure2.png", fig)