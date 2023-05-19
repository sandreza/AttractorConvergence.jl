levels = round(Int, log2(maximum(union(markov_chain))))
level_list = 1:levels
eigenvalues_list = Vector{ComplexF64}[]
right_eigenvectors_list = Array{ComplexF64}[]
left_eigenvectors_list = Array{ComplexF64}[]
reversible_eigenvalues_list = Vector{Float64}[]
rotational_eigenvalues_list = Vector{ComplexF64}[]
probabilities_list = Vector{Float64}[]
for level in ProgressBar(level_list)
    markov_states = get_markov_states(centers_list, level)
    coarse_grained_markov_chain = div.(markov_chain .- 1, 2^(levels - level)) .+ 1
    coarse_grained_markov_chain_2 = div.(s_markov_chain .- 1, 2^(levels - level)) .+ 1
    Q1 = BayesianGenerator(coarse_grained_markov_chain; dt=Δt)
    Q2 = BayesianGenerator(coarse_grained_markov_chain_2, Q1.posterior; dt=Δt)
    Q = mean(Q2)
    Λ, V = eigen(Q)
    p = real.(V[:, end] ./ sum(V[:, end])) # steady state distribution / invariant measure
    push!(probabilities_list, p)
    push!(eigenvalues_list, Λ)
    push!(right_eigenvectors_list, V)
    push!(left_eigenvectors_list, inv(V))

    # Reversible part of the operator
    Q̃ = Symmetric(Diagonal(1 ./ sqrt.(p)) * Q * Diagonal(sqrt.(p)) + Diagonal(sqrt.(p)) * Q' * Diagonal(1 ./ sqrt.(p))) ./ 2
    Λ̃, Ṽ = eigen(Q̃)
    push!(reversible_eigenvalues_list, Λ̃)
    # Rotational part of the operator
    Q̂ = ( Diagonal(1 ./ sqrt.(p)) * Q * Diagonal(sqrt.(p)) - Diagonal(sqrt.(p)) * Q' * Diagonal(1 ./ sqrt.(p)) ) ./ 2
    Λ̂, V̂ = eigen(Q̂)
    push!(rotational_eigenvalues_list, Λ̂)
end