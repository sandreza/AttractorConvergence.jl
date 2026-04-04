using MarkovChainHammer


hfile = h5open(data_directory * "eigenvalues.hdf5", "r")

ps = Vector{Float64}[]
for i in 1:25
    push!(ps, read(hfile["generator steady state $i"]))
end

close(hfile)

entropies = scaled_entropy.(ps)
cells = length.(ps)

fig = Figure(size = (650, 400), fontsize = 20)
ax = Axis(fig[1, 1]; xlabel = L"\text{Cells}", 
                     ylabel = L"\text{Scaled entropy}", 
                     xticks = ([2, 4, 6], [L"10^2", L"10^4", L"10^6"]),
                     yticks = ([0.98, 0.99, 1.0, 1.01], [L"0.98", L"0.99", L"1.00", L"1.01"])) # title = L"\text{Scaled Entropy vs. Cells}")
scatterlines!(ax, log10.(cells), entropies, markersize = 8, linewidth = 0.5, linecolor = :grey, color = :black, label = "data")
hlines!(ax, 1, color = :red, linestyle = :dash)
text!(ax, 2.5, 1.0008, text = L"\text{Entropy of a uniform distribution}", color = :red, rotation = 0, fontsize = 15)
ylims!(ax, 0.975, 1.003)
xlims!(ax, 0.5, 6.1)

save(figure_directory * "/Figure3.eps", fig)
save(figure_directory * "/Figure3.png", fig)