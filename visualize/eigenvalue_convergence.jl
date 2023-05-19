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