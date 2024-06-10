using StateSpacePartitions, MarkovChainHammer

include("utils.jl")

hfile = h5open(data_directory  * "/embedding.hdf5", "r")
markov_chain = read(hfile["markov_chain"])
coarse_markov_chain = read(hfile["coarse_markov_chains"])
close(hfile)

@info "loading data"
hfile = h5open(data_directory  * "/lorenz.hdf5", "r")
m_timeseries = read(hfile["timeseries"])
s_timeseries = read(hfile["symmetrized timeseries"])
dt = read(hfile["dt"])
joined_timeseries = hcat(m_timeseries, s_timeseries) # only for Partitioning Purpose
close(hfile)

@info "opening centers"
centerslist = []
hfile = h5open(data_directory  * "/centers.hdf5", "r")
for i in ProgressBar(eachindex(coarse_probabilities))
    centers = read(hfile["centers $i"])
    push!(centerslist, centers)
end
close(hfile)

N = size(coarse_markov_chain)[1]
N2 = N ÷ 2

observable(s) = s[3]

Tfinal = 40
numsteps = ceil(Int, Tfinal / dt)
runge_kutta_correlation = zeros(numsteps)
dt_perron_frobenius_correlation = zeros(numsteps)

ztruth = autocovariance(m_timeseries[3, 1:10^6]; timesteps = numsteps, progress = true)
ztruth_s = autocovariance(s_timeseries[3, 1:10^6]; timesteps = numsteps, progress = true)
ztruth = (ztruth + ztruth_s)/2

runge_kutta_correlations = []
dt_perron_frobenius_correlations = []
for i in 1:14
    partitions = coarse_markov_chain[1:N2, i]
    partitions_s = coarse_markov_chain[1+N2:end, i]

    # Generator
    Q = sparse_generator(partitions; dt = dt)
    Q = (Q + sparse_generator(partitions_s; dt = dt))/2
    𝒪 = [observable(markov_state) for markov_state in eachcol(centerslist[i])]
    guess_p = steady_state(partitions)
    refine_p, _ = inverse_iteration(Q, guess_p, 1e-2; tol = 1e-5, maxiter_eig = 20, maxiter_solve = 3, τ = 0.1)
    p = refine_p ./ sum(refine_p)
    println("the error for $i is ", maximum(abs.(Q * p) / maximum(p)))
    sQ = SparseGenerator(Q', dt);
    rk4 = RungeKutta4(length(p))
    observable_trajectory = copy(𝒪)
    runge_kutta_correlation[1] = sum(𝒪 .* p .* observable_trajectory) .- sum(p .* 𝒪)^2
    @info "runge-kutta autocovariance"
    for i in ProgressBar(2:numsteps)
        rk4(sQ, observable_trajectory, dt)
        observable_trajectory .= rk4.xⁿ⁺¹
        runge_kutta_correlation[i] = sum(𝒪 .* p .* observable_trajectory) .- sum(p .* 𝒪)^2
    end
    push!(runge_kutta_correlations, copy(runge_kutta_correlation))

    # Perron-Frobenius for Δt
    P = sparse_perron_frobenius(partitions)
    P = (P + sparse_perron_frobenius(partitions_s))/2
    refine_p, _ = inverse_iteration(P, refine_p, 1+1e-2; tol = 1e-5, maxiter_eig = 20, maxiter_solve = 3, τ = 0.1)
    p = refine_p ./ sum(refine_p)

    observable_trajectory = copy(𝒪)
    dt_perron_frobenius_correlation[1] = sum(𝒪 .* p .* observable_trajectory) .- sum(p .* 𝒪)^2
    @info "perron-frobenius autocovariance"
    for i in ProgressBar(2:numsteps)
        observable_trajectory .= P' * observable_trajectory
        dt_perron_frobenius_correlation[i] = sum(𝒪 .* p .* observable_trajectory) .- sum(p .* 𝒪)^2
    end
    push!(dt_perron_frobenius_correlations, copy(dt_perron_frobenius_correlation))
end