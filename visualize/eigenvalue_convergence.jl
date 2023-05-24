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
##
fig_eig = Figure(resolution=(1000, 500))

ax = Axis(fig_eig[1, 1]; xlabel="# of partitions", ylabel="real(λ)", title="Real Part of Eigenvalues")
λs = Vector{Float64}[]
lastfew = 7
ps = 9
for j in 1:lastfew
    tmp = [real.(eigenvalues_list[end-i][end-j]) for i in 0:ps]
    push!(λs, tmp)
end
colors = [:red, :blue, :orange, :green, :purple, :pink, :yellow, :cyan, :magenta]
labels = ["Eigenvalue $i" for i in 1:lastfew]
for i in eachindex(λs)
    λs1 = λs[i]
    lines!(ax, [2 .^(levels .- i) for i in 0:ps], λs1; color=colors[i], label=labels[i])
    scatter!(ax, [2 .^ (levels .- i) for i in 0:ps], λs1; color=colors[i])
end
axislegend(ax, position = :rb)
xlims!(ax, 0, 4200)
ylims!(ax, -15, 0)

ax = Axis(fig_eig[1, 2]; xlabel="# of partitions", ylabel="imag(λ)", title="Imaginary Part of Eigenvalues")
λs = Vector{Float64}[]
lastfew = 7
ps = 9
for j in 1:lastfew
    tmp = [imag.(eigenvalues_list[end-i][end-j]) for i in 0:ps]
    push!(λs, tmp)
end
colors = [:red, :blue, :orange, :green, :purple, :pink, :yellow, :cyan, :magenta]
labels = ["Eigenvalue $i" for i in 1:lastfew]
for i in eachindex(λs)
    λs1 = λs[i]
    lines!(ax, [2 .^ (levels .- i) for i in 0:ps], λs1; color=colors[i], label=labels[i])
    scatter!(ax, [2 .^ (levels .- i) for i in 0:ps], λs1; color=colors[i])
end
# axislegend(ax, position=:rb)
xlims!(ax, 0, 4200)
display(fig_eig)

##
#=
ax = Axis(fig_eig[2, 1]; xlabel="# of partitions", ylabel="λ", title="Reversible Eigenvalues")
λs = Vector{Float64}[]
lastfew = 7
ps = 9
for j in 1:lastfew
    tmp = [real.(reversible_eigenvalues_list[end-i][end-j]) for i in 0:ps]
    push!(λs, tmp)
end
colors = [:red, :blue, :orange, :green, :purple, :pink, :yellow, :cyan, :magenta]
labels = ["Eigenvalue $i" for i in 1:lastfew]
for i in eachindex(λs)
    λs1 = λs[i]
    lines!(ax, [2 .^ (levels .- i) for i in 0:ps], λs1; color=colors[i], label=labels[i])
    scatter!(ax, [2 .^ (levels .- i) for i in 0:ps], λs1; color=colors[i])
end
# axislegend(ax, position=:rb)
ylims!(ax, -5, 0)
xlims!(ax, 0, 4200)
=#

#=
ax = Axis(fig_eig[2, 2]; xlabel="# of partitions", ylabel="imag(λ)", title="Imaginary Part of Rotational Eigenvalues")
λs = Vector{Float64}[]
lastfew = 7
ps = 9
for j in 1:lastfew
    tmp = [imag.(rotational_eigenvalues_list[end-i][end-j]) for i in 0:ps]
    tmp2 = sort(tmp)
    tmp3 = tmp2[tmp2 .> 0 ][2:lastfew+1]
    push!(λs, tmp3)
end
colors = [:red, :blue, :orange, :green, :purple, :pink, :yellow, :cyan, :magenta]
labels = ["Eigenvalue $i" for i in 1:lastfew]
for i in eachindex(λs)
    λs1 = λs[i]
    lines!(ax, [2 .^ (levels .- i) for i in 0:ps], λs1; color=colors[i], label=labels[i])
    scatter!(ax, [2 .^ (levels .- i) for i in 0:ps], λs1; color=colors[i])
end
axislegend(ax, position=:rc)
xlims!(ax, 0, 4200)



display(fig_eig)
##
ax = Axis(fig_eig[1, 2]; xlabel="log2(# of partitions)", ylabel="log2(|imag(λ)|)", title="Imaginary Part of Eigenvalues")
lastfew = 9
xlims!(ax, (levels - lastfew, levels + 1))
ylims!(ax, (-1, 5))
colors = [:red, :blue, :orange, :green, :purple, :pink, :yellow, :cyan, :magenta]

for i in 0:8
    λs1 = log2.(abs.(imag.(eigenvalues_list[end-i][end-lastfew:end-1])))
    scatter!(ax, levels .- i .+ zeros(lastfew), λs1; color=colors)
end

ax = Axis(fig_eig[2, 1]; xlabel="log2(# of partitions)", ylabel="log2(|λ|)", title="Reversible Eigenvalues")
lastfew = 9
xlims!(ax, (levels - lastfew, levels + 1))
colors = [:red, :blue, :orange, :green, :purple, :pink, :yellow, :cyan, :magenta]
for i in 0:8
    λs1 = log2.(-reversible_eigenvalues_list[end-i][end-lastfew:end-1])
    scatter!(ax, levels .- i .+ zeros(lastfew), λs1; color=colors)
end
ax = Axis(fig_eig[2, 2]; xlabel="log2(# of partitions)", ylabel="log2(|λ|)", title="Rotational Eigenvalues")
xlims!(ax, (levels - lastfew, levels + 1))
for i in 0:7
    λs = sort(imag.(rotational_eigenvalues_list[end-i]))
    λs = λs[λs.>0][2:lastfew+1]
    println(λs)
    λs1 = log2.(λs)
    scatter!(ax, levels .- i .+ zeros(lastfew), λs1; color=reverse(colors))
end
display(fig_eig)
=#