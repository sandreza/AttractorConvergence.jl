using HDF5, ProgressBars, MarkovChainHammer, AttractorConvergence, SparseArrays

include("utils.jl")

data_directory = "/real_data"

@info "loading data"
mcfile = h5open(pwd() * data_directory  * "/embedding.hdf5", "r")
centers_hfile = h5open(pwd() * data_directory  * "/centers.hdf5", "r")
eigenvalues_hfile = h5open(pwd() * data_directory  * "/eigenvalues.hdf5", "r")

generator_compute = true # increase computation by a factor of 4

coarse_markov_chain = read(mcfile["coarse_markov_chains 1"])
dt = read(eigenvalues_hfile["generator dt 1"])
imax = length(keys(centers_hfile)) - 1

close(mcfile)
close(centers_hfile)
close(eigenvalues_hfile)

# hfile = h5open(pwd() * data_directory  * "/temporal_autocovariance.hdf5", "w")
Tfinal = 10
numsteps = ceil(Int, Tfinal / dt) + 1
q_runge_kutta_correlation = zeros(numsteps)
pf1_runge_kutta_correlation = zeros(numsteps)
numsteps10 = (numsteps-1)÷10 + 1
pf10_runge_kutta_correlation = zeros(numsteps10)
numsteps100 = (numsteps-1)÷100 + 1
pf100_runge_kutta_correlation = zeros(numsteps100)

@info "starting loop"
# something strange happened with 7, so we skip it
for i in ProgressBar(18:imax)
    println("On case $i")
    mcfile = h5open(pwd() * data_directory  * "/embedding.hdf5", "r")
    centers_hfile = h5open(pwd() * data_directory  * "/centers.hdf5", "r")
    eigenvalues_hfile = h5open(pwd() * data_directory  * "/eigenvalues.hdf5", "r")

    coarse_markov_chain = read(mcfile["coarse_markov_chains $i"])
    centers = read(centers_hfile["centers $i"])
    p = read(eigenvalues_hfile["generator steady state $i"] )
    p1 = read(eigenvalues_hfile["perron_frobenius 1 steady state $i"] )
    close(mcfile)
    close(centers_hfile)
    close(eigenvalues_hfile)


    N = length(coarse_markov_chain)
    N2 = N ÷ 2

    @info "calculating operators"
    if length(p) < 4e4
        if generator_compute
            Qa = generator(coarse_markov_chain[1:N2]; dt = dt)
            Qb = generator(coarse_markov_chain[N2+1:end]; dt = dt)
            Q = (Qa + Qb)/2
        else
            PF1a = perron_frobenius(coarse_markov_chain[1:N2]; step = 1)
            PF1b = perron_frobenius(coarse_markov_chain[N2+1:end]; step = 1)
            PF1 = (PF1a + PF1b)/2
        end
    else
        if generator_compute
            Qa = sparse_generator(coarse_markov_chain[1:N2]; dt = dt)
            Qb = sparse_generator(coarse_markov_chain[N2+1:end]; dt = dt)
            Q = (Qa + Qb)/2
        else
            PF1a = sparse_perron_frobenius(coarse_markov_chain[1:N2]; step = 1)
            PF1b = sparse_perron_frobenius(coarse_markov_chain[N2+1:end]; step = 1)
            PF1 = (PF1a + PF1b)/2
        end
    end

    @info "calculating observable"
    𝒪 = [center[3] for center in eachcol(centers)]

    if generator_compute
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
        hfile = h5open(pwd() * data_directory  * "/temporal_autocovariance.hdf5", "r+")
        hfile["ensemble mean autocovariance generator $i"] = q_runge_kutta_correlation
    else
        @info "perron-frobenius 1 autocovariance"
        observable_trajectory = copy(𝒪)
        pf1_runge_kutta_correlation[1] = sum(𝒪 .* p1 .* observable_trajectory) .- sum(p1 .* 𝒪)^2
        for jj in ProgressBar(2:numsteps)
            observable_trajectory .= PF1' * observable_trajectory
            pf1_runge_kutta_correlation[jj] = sum(𝒪 .* p1 .* observable_trajectory) .- sum(p1 .* 𝒪)^2
        end
        hfile = h5open(pwd() * data_directory  * "/temporal_autocovariance.hdf5", "r+")
        hfile["ensemble mean autocovariance perron_frobenius 1 $i"] = pf1_runge_kutta_correlation
    end
    close(hfile)
end