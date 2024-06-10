using HDF5, MarkovChainHammer, StateSpacePartitions

data_directory = "/storage4/andre/attractor_convergence" * "/real_data"

@info "opening lorenz file"
hfile = h5open(data_directory  * "/lorenz.hdf5", "r")
dt = read(hfile["dt"])
close(hfile)



@info "opening embedding file"
nhfile = h5open(data_directory  * "/embedding.hdf5", "r")
@info "opening embedding data 20"
coarse_markov_chain_20 = read(nhfile["coarse_markov_chains 20"])
@info "opening embedding data 22"
coarse_markov_chain_22 = read(nhfile["coarse_markov_chains 22"])
@info "opening embedding data 25"
coarse_markov_chain_25 = read(nhfile["coarse_markov_chains 25"])
close(nhfile)

@info "reading moments"
hfile = h5open(data_directory  * "/ensemble_mean_statistics.hdf5", "r")
z_ensemble_20 = read(hfile["z moments 20"])
z_ensemble_22 = read(hfile["z moments 22"])
z_ensemble_25 = read(hfile["z moments 25"])
close(hfile)


@info "reading centers"
centers_hfile = h5open(data_directory  * "/centers.hdf5", "r")
centers_20 = read(centers_hfile["centers 20"])
centers_22 = read(centers_hfile["centers 22"])
centers_25 = read(centers_hfile["centers 25"])
close(centers_hfile)

@info "reading time mean statistics"
hfile = h5open(data_directory  * "/time_mean_statistics.hdf5", "r")
z_moments_time = read(hfile["z moments"])
close(hfile)

@info "calculating relative error"
relative_error_20 = abs.(z_ensemble_20 - z_moments_time) ./ z_moments_time
relative_error_22 = abs.(z_ensemble_22 - z_moments_time) ./ z_moments_time
relative_error_25 = abs.(z_ensemble_25 - z_moments_time) ./ z_moments_time


@info "computing higher tolerance case 20"
coarse_markov_chain = coarse_markov_chain_20
N = length(coarse_markov_chain)
N2 = N ÷ 2
Q1 = sparse_generator(coarse_markov_chain[1:N2]; dt = dt)
Q2 = sparse_generator(coarse_markov_chain[N2+1:end]; dt = dt)
Q20 = (Q1 + Q2)/2

guess_p = ones(size(Q20)[1])
refine_p, _ = inverse_iteration(Q20, guess_p, 1e-3; tol = 1e-6, maxiter_eig = 40, maxiter_solve = 5, τ = 0.1)
p20 = refine_p ./ sum(refine_p)

z_ensemble_20_new = length(z_ensemble_20)
observables = [i -> i[3]^j for j in 1:5]
centers = centers_20
partition_number = size(centers)[2]

for (oo,observable) in enumerate(observables)
    for ii in 1:partition_number
        z_ensemble_20_new[oo] += observable(centers[:, ii]) .* p20[ii]
    end
end


@info "computing higher tolerance case 22"
coarse_markov_chain = coarse_markov_chain_22
N2 = N ÷ 2
Q1 = sparse_generator(coarse_markov_chain[1:N2]; dt = dt)
Q2 = sparse_generator(coarse_markov_chain[N2+1:end]; dt = dt)
Q22 = (Q1 + Q2)/2

guess_p = ones(size(Q22)[1])
refine_p, _ = inverse_iteration(Q22, guess_p, 1e-3; tol = 1e-6, maxiter_eig = 40, maxiter_solve = 5, τ = 0.1)
p22 = refine_p ./ sum(refine_p)

z_ensemble_22_new = length(z_ensemble_22)
observables = [i -> i[3]^j for j in 1:5]
centers = centers_22
partition_number = size(centers)[2]

for (oo,observable) in enumerate(observables)
    for ii in 1:partition_number
        z_ensemble_22_new[oo] += observable(centers[:, ii]) .* p22[ii]
    end
end

@info "computing higher tolerance case 25"
coarse_markov_chain = coarse_markov_chain_25
N2 = N ÷ 2
Q1 = sparse_generator(coarse_markov_chain[1:N2]; dt = dt)
Q2 = sparse_generator(coarse_markov_chain[N2+1:end]; dt = dt)
Q25 = (Q1 + Q2)/2

guess_p = ones(size(Q25)[1])
refine_p, _ = inverse_iteration(Q25, guess_p, 1e-3; tol = 1e-6, maxiter_eig = 40, maxiter_solve = 5, τ = 0.1)
p25 = refine_p ./ sum(refine_p)

z_ensemble_25_new = length(z_ensemble_25)
observables = [i -> i[3]^j for j in 1:5]
centers = centers_25
partition_number = size(centers)[2]

for (oo,observable) in enumerate(observables)
    for ii in 1:partition_number
        z_ensemble_25_new[oo] += observable(centers[:, ii]) .* p25[ii]
    end
end

