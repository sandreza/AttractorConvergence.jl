using GLMakie, AttractorConvergence, SparseArrays
import MarkovChainHammer.Utils: histogram
using MarkovChainHammer.BayesianMatrix
using MarkovChainHammer.TransitionMatrix: entropy

fig = Figure(resolution = (2000,2000))
for ii in 1:9
    i = div(ii-1, 3) + 1
    j = mod(ii-1, 3) + 1
    start_value = ii
    ax = Axis(fig[j, i]; title="ensemble level $(start_value) and $(start_value + 1)", xlabel="real", ylabel="imaginary")
    eigenlist = eigenvalues_list[start_value]
    scatter!(ax, real.(eigenlist), imag.(eigenlist), color=(:red, 0.5), label="level $(start_value)")
    eigenlist = eigenvalues_list[start_value+1]
    scatter!(ax, real.(eigenlist), imag.(eigenlist), color=(:blue, 0.5), label="level $(start_value + 1)")
    axislegend(ax)
end
display(fig)

fig_eig = Figure(resolution=(1200, 400))

ax = Axis(fig_eig[1, 1]; xlabel="log2(# of partitions)", ylabel="log2(|λ|)", title="Real Part of Eigenvalues")
lastfew = 9
xlims!(ax, (levels - lastfew, levels + 1))
colors = [:red, :blue, :orange, :green, :purple, :pink, :yellow, :cyan, :magenta]
for i in 0:8
    λs1 = log2.(-real.(eigenvalues_list[end-i][end-lastfew:end-1]))
    scatter!(ax, levels .- i .+ zeros(lastfew), λs1; color=colors)
end

ax = Axis(fig_eig[1, 2]; xlabel="log2(# of partitions)", ylabel="log2(|λ|)", title="Reversible Eigenvalues")
lastfew = 9
xlims!(ax, (levels - lastfew, levels + 1))
colors = [:red, :blue, :orange, :green, :purple, :pink, :yellow, :cyan, :magenta]
for i in 0:8
    λs1 = log2.(-reversible_eigenvalues_list[end-i][end-lastfew:end-1])
    scatter!(ax, levels .- i .+ zeros(lastfew), λs1; color=colors)
end
ax = Axis(fig_eig[1, 3]; xlabel="log2(# of partitions)", ylabel="log2(|λ|)", title="Rotational Eigenvalues")
xlims!(ax, (levels - lastfew, levels + 1))
for i in 0:7
    λs = sort(imag.(rotational_eigenvalues_list[end-i]))
    λs = λs[λs.>0][2:lastfew+1]
    println(λs)
    λs1 = log2.(λs)
    scatter!(ax, levels .- i .+ zeros(lastfew), λs1; color=reverse(colors))
end
display(fig_eig)