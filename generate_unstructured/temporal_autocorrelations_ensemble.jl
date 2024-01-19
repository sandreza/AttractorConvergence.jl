using HDF5, ProgressBars, MarkovChainHammer, AttractorConvergence, SparseArrays

include("utils.jl")

data_directory = "/real_data"

@info "loading data"
mcfile = h5open(pwd() * data_directory  * "/embedding.hdf5", "r")
centers_hfile = h5open(pwd() * data_directory  * "/centers.hdf5", "r")
eigenvalues_hfile = h5open(pwd() * data_directory  * "/eigenvalues.hdf5", "r")


coarse_markov_chain = read(mcfile["coarse_markov_chains 1"])
dt = read(eigenvalues_hfile["generator dt 1"])
imax = length(keys(centers_hfile)) - 1

close(mcfile)
close(centers_hfile)
close(eigenvalues_hfile)

Tfinal = 40
numsteps = ceil(Int, Tfinal / dt) + 1
q_runge_kutta_correlation = zeros(numsteps)
pf1_runge_kutta_correlation = zeros(numsteps)
numsteps10 = (numsteps-1)÷10 + 1
pf10_runge_kutta_correlation = zeros(numsteps10)
numsteps100 = (numsteps-1)÷100 + 1
pf100_runge_kutta_correlation = zeros(numsteps100)

@info "starting loop"
# something strange happened with 7, so we skip it
for i in ProgressBar(8:imax)
    println("On case $i")
    mcfile = h5open(pwd() * data_directory  * "/embedding.hdf5", "r")
    centers_hfile = h5open(pwd() * data_directory  * "/centers.hdf5", "r")
    eigenvalues_hfile = h5open(pwd() * data_directory  * "/eigenvalues.hdf5", "r")

    coarse_markov_chain = read(mcfile["coarse_markov_chains $i"])
    centers = read(centers_hfile["centers $i"])
    p = read(eigenvalues_hfile["generator steady state $i"] )
    p1 = read(eigenvalues_hfile["perron_frobenius 1 steady state $i"] )
    p10 = read(eigenvalues_hfile["perron_frobenius 10 steady state $i"] )
    p100 = read(eigenvalues_hfile["perron_frobenius 100 steady state $i"] )

    close(mcfile)
    close(centers_hfile)
    close(eigenvalues_hfile)

    hfile = h5open(pwd() * data_directory  * "/temporal_autocovariance.hdf5", "r+")

    N = length(coarse_markov_chain)
    N2 = N ÷ 2

    @info "calculating operators"
    if length(p) < 4e4
        Qa = generator(coarse_markov_chain[1:N2]; dt = dt)
        Qb = generator(coarse_markov_chain[N2+1:end]; dt = dt)
        Q = (Qa + Qb)/2
        PF1a = perron_frobenius(coarse_markov_chain[1:N2]; step = 1)
        PF1b = perron_frobenius(coarse_markov_chain[N2+1:end]; step = 1)
        PF1 = (PF1a + PF1b)/2
        PF10a = perron_frobenius(coarse_markov_chain[1:N2]; step = 10)
        PF10b = perron_frobenius(coarse_markov_chain[N2+1:end]; step = 10)
        PF10 = (PF10a + PF10b)/2
        PF100a = perron_frobenius(coarse_markov_chain[1:N2]; step = 100)
        PF100b = perron_frobenius(coarse_markov_chain[N2+1:end]; step = 100)
        PF100 = (PF100a + PF100b)/2
    else
        Qa = sparse_generator(coarse_markov_chain[1:N2]; dt = dt)
        Qb = sparse_generator(coarse_markov_chain[N2+1:end]; dt = dt)
        Q = (Qa + Qb)/2
        PF1a = sparse_perron_frobenius(coarse_markov_chain[1:N2]; step = 1)
        PF1b = sparse_perron_frobenius(coarse_markov_chain[N2+1:end]; step = 1)
        PF1 = (PF1a + PF1b)/2
        PF10a = sparse_perron_frobenius(coarse_markov_chain[1:N2]; step = 10)
        PF10b = sparse_perron_frobenius(coarse_markov_chain[N2+1:end]; step = 10)
        PF10 = (PF10a + PF10b)/2
        PF100a = sparse_perron_frobenius(coarse_markov_chain[1:N2]; step = 100)
        PF100b = sparse_perron_frobenius(coarse_markov_chain[N2+1:end]; step = 100)
        PF100 = (PF100a + PF100b)/2
    end

    @info "calculating observable"
    𝒪 = [center[3] for center in eachcol(centers)]

    sQ = SparseGenerator(Q', dt);
    rk4 = RungeKutta4(length(p))
    observable_trajectory = copy(𝒪)
    q_runge_kutta_correlation[1] = sum(𝒪 .* p .* observable_trajectory) .- sum(p .* 𝒪)^2
    @info "generator runge-kutta autocovariance"
    for jj in ProgressBar(2:numsteps)
        rk4(sQ, observable_trajectory, dt)
        observable_trajectory .= rk4.xⁿ⁺¹
        q_runge_kutta_correlation[jj] = sum(𝒪 .* p .* observable_trajectory) .- sum(p .* 𝒪)^2
    end
    hfile["ensemble mean autocovariance generator $i"] = q_runge_kutta_correlation
    @info "perron-frobenius 1 autocovariance"

    observable_trajectory = copy(𝒪)
    pf1_runge_kutta_correlation[1] = sum(𝒪 .* p1 .* observable_trajectory) .- sum(p1 .* 𝒪)^2
    for jj in ProgressBar(2:numsteps)
        observable_trajectory .= PF1' * observable_trajectory
        pf1_runge_kutta_correlation[jj] = sum(𝒪 .* p1 .* observable_trajectory) .- sum(p1 .* 𝒪)^2
    end
    hfile["ensemble mean autocovariance perron_frobenius 1 $i"] = pf1_runge_kutta_correlation

    @info "perron-frobenius 10 autocovariance"
    observable_trajectory = copy(𝒪)
    pf10_runge_kutta_correlation[1] = sum(𝒪 .* p10 .* observable_trajectory) .- sum(p10 .* 𝒪)^2
    for jj in ProgressBar(2:numsteps10)
        observable_trajectory .= PF10' * observable_trajectory
        pf10_runge_kutta_correlation[jj] = sum(𝒪 .* p10 .* observable_trajectory) .- sum(p10 .* 𝒪)^2
    end
    hfile["ensemble mean autocovariance perron_frobenius 10 $i"] = pf10_runge_kutta_correlation

    @info "perron-frobenius 100 autocovariance"
    observable_trajectory = copy(𝒪)
    pf100_runge_kutta_correlation[1] = sum(𝒪 .* p100 .* observable_trajectory) .- sum(p100 .* 𝒪)^2
    for jj in ProgressBar(2:numsteps100)
        observable_trajectory .= PF100' * observable_trajectory
        pf100_runge_kutta_correlation[jj] = sum(𝒪 .* p100 .* observable_trajectory) .- sum(p100 .* 𝒪)^2
    end
    hfile["ensemble mean autocovariance perron_frobenius 100 $i"] = pf100_runge_kutta_correlation
    close(hfile)
end