using GLMakie, AttractorConvergence, SparseArrays
import MarkovChainHammer.Utils: histogram
using MarkovChainHammer.BayesianMatrix
using MarkovChainHammer.TransitionMatrix: entropy

@info "loading data"
hfile = h5open(pwd() * "/data/embedding.hdf5", "r")
markov_chain = read(hfile["markov_chain"])
s_markov_chain = read(hfile["symmetrized markov chain"])
Δt = read(hfile["dt"])
close(hfile)
hfile = h5open(pwd() * "/data/lorenz.hdf5", "r")
timeseries = read(hfile["timeseries"])
s_timeseries = read(hfile["symmetrized timeseries"])
close(hfile)
# read centers list from file 
hfile = h5open(pwd() * "/data/kmeans.hdf5", "r")
centers_matrix = read(hfile["centers"])
levels = read(hfile["levels"])
close(hfile)
centers_list = [[centers_matrix[:, 1, i], centers_matrix[:, 2, i]] for i in 1:size(centers_matrix)[3]]
##
embedding = StateTreeEmbedding(centers_list, levels)
##
function get_markov_states(centers_list::Vector{Vector{Vector{Float64}}}, level)
    markov_states = Vector{Float64}[]
    indices = level_global_indices(level)
    for index in indices
        push!(markov_states, centers_list[index][1])
        push!(markov_states, centers_list[index][2])
    end
    return markov_states
end
function sparsify(Q; threshold=eps(10^6.0))
    Q[abs.(Q).<threshold] .= 0
    return sparse(Q)
end
##
# Need consistency between centers and markov_chain
observable(state) = state[3]
gₜ = [observable(timeseries[:, i]) for i in 1:size(timeseries)[2]]
level_list = 1:levels
time_moment = mean(gₜ)
observable_lists = Vector{Float64}[]
probabilities_list = Vector{Float64}[]
observable_values = Float64[]
sparsity_list = Float64[]
eigenvalues_list = Vector{ComplexF64}[]
for level in ProgressBar(level_list)
    markov_states = get_markov_states(centers_list, level)
    coarse_grained_markov_chain = div.(markov_chain .- 1, 2^(levels - level)) .+ 1
    coarse_grained_markov_chain_2 = div.(s_markov_chain .- 1, 2^(levels - level)) .+ 1
    Q1 = BayesianGenerator(coarse_grained_markov_chain; dt=Δt)
    Q2 = BayesianGenerator(coarse_grained_markov_chain_2, Q1.posterior; dt=Δt)
    Q = mean(Q2)
    Λ, V = eigen(Q)
    push!(eigenvalues_list, Λ)
    sQ = sparsify(Q)
    sparsity = length(sQ.nzval) / length(sQ)
    push!(sparsity_list, sparsity)
    p = steady_state(Q)
    push!(probabilities_list, p)
    gₑ = observable.(markov_states)
    push!(observable_lists, gₑ)
    observable_value = sum(gₑ .* p)
    push!(observable_values, observable_value)
end
##
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