
level_list = 1:levels
eigenvalues_list = Vector{ComplexF64}[]
right_eigenvectors_list = Array{ComplexF64}[]
left_eigenvectors_list = Array{ComplexF64}[]
for level in ProgressBar(level_list)
    markov_states = get_markov_states(centers_list, level)
    coarse_grained_markov_chain = div.(markov_chain .- 1, 2^(levels - level)) .+ 1
    coarse_grained_markov_chain_2 = div.(s_markov_chain .- 1, 2^(levels - level)) .+ 1
    Q1 = BayesianGenerator(coarse_grained_markov_chain; dt=Δt)
    Q2 = BayesianGenerator(coarse_grained_markov_chain_2, Q1.posterior; dt=Δt)
    Q = mean(Q2)
    Λ, V = eigen(Q)
    push!(eigenvalues_list, Λ)
    push!(right_eigenvectors_list, V)
    push!(left_eigenvectors_list, inv(V))
end

##
using GLMakie 

fig = Figure(resolution = (800, 600))
ax = Axis(fig[1, 1])
levellist = 3:levels
λs1 = [real(eigenvalues_list[level][end-1]) for level in levellist]
scatter!(ax, levellist, λs1, color=(:red, 0.5))
λs2 = [real(eigenvalues_list[level][end-2]) for level in levellist]
scatter!(ax, levellist, λs2, color=(:blue, 0.5))
λs3 = [real(eigenvalues_list[level][end-3]) for level in levellist]
scatter!(ax, levellist, λs3, color=(:green, 0.5))
λs4 = [real(eigenvalues_list[level][end-4]) for level in levellist]
scatter!(ax, levellist, λs4, color=(:orange, 0.5))
display(fig)

#=
for level in 2:levels 

    λ = eigenvalues_list[level][end-1]
    println(level)
    println(real.(λ))
    scatter!(ax, [level], [real.(λ)], color = :red)
    # scatter!(ax, level, real.(eigenvalues_list[level][end-1]), color = :blue)
    # scatter!(ax, level, real.(eigenvalues_list[level][end-2]), color = :orange)
end
display(fig)
=#