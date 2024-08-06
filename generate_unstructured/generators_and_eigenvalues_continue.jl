using HDF5, MarkovChainHammer, ProgressBars, LinearAlgebra, Statistics, Random, SparseArrays
using StateSpacePartitions

# data_directory = "/storage4/andre/attractor_convergence" * "/real_data"

first_index = 1
hfile = h5open(data_directory  * "/lorenz.hdf5", "r")
dt = read(hfile["dt"])
close(hfile)

hfile = h5open(data_directory  * "/embedding.hdf5", "r")
coarse_markov_chain = read(hfile["coarse_markov_chains $first_index"])
probability = read(hfile["probability"])
coarse_probabilities = read(hfile["coarse_probabilities"])
close(hfile)

hfile = h5open(data_directory  * "/eigenvalues.hdf5", "r")
tmp = [parse(Int, key[end-1:end]) for key in keys(hfile)]
last_index = maximum(tmp)
close(hfile)
println("Picking up at index $last_index")

for (old_index, probability) in ProgressBar(enumerate(coarse_probabilities[last_index+1:end]))
    hfile = h5open(data_directory  * "/eigenvalues.hdf5", "r+")
    index = last_index + old_index
    nhfile = h5open(data_directory  * "/embedding.hdf5", "r")
    coarse_markov_chain = read(nhfile["coarse_markov_chains $index"])
    close(nhfile)
    N = length(coarse_markov_chain)
    N2 = N ÷ 2
    Q1 = sparse_generator(coarse_markov_chain[1:N2]; dt = dt)
    Q2 = sparse_generator(coarse_markov_chain[N2+1:end]; dt = dt)
    Q = (Q1 + Q2)/2
    ##
    guess_p = ones(size(Q)[1])
    refine_p, _ = inverse_iteration(Q, guess_p, 1e-2; tol = 1e-5, maxiter_eig = 20, maxiter_solve = 3, τ = 0.1)
    p = refine_p ./ sum(refine_p)
    ##
    hfile["generator steady state $index"] = p
    hfile["generator dt $index"] = dt
    ##
    guess_v = rand(size(Q)[1])
    guess_λ = read(hfile["generator koopman eigenvalue $(index-1)"])
    w, λ = inverse_iteration(Q', guess_v, guess_λ; tol = 1e-5, maxiter_eig = 20, maxiter_solve = 3, τ = 0.1)
    w = w / norm(w)
    ##
    hfile["generator koopman eigenvalue $index"] = λ
    hfile["generator koopman eigenvector $index"] = copy(w)

    for k in [1, 10, 100]
        P1 = sparse_perron_frobenius(coarse_markov_chain[1:N2]; step = k)
        P2 = sparse_perron_frobenius(coarse_markov_chain[N2+1:end]; step = k)
        P = (P1 + P2)/2
        ##
        refine_p, _ = inverse_iteration(P, copy(p), 1+1e-2; tol = 1e-5, maxiter_eig = 20, maxiter_solve = 3, τ = 0.1)
        p = refine_p ./ sum(refine_p)
        hfile["perron_frobenius $k steady state $index"] = p
        hfile["perron_frobenius $k dt $index"] = k * dt
        ##
        guess_λ = read(hfile["perron_frobenius $k koopman eigenvalue $(index-1)"])
        w, λ = inverse_iteration(P', copy(w), guess_λ; tol = 1e-5, maxiter_eig = 20, maxiter_solve = 3, τ = 0.1)
        w = w /norm(w)
        hfile["perron_frobenius $k koopman eigenvalue $index"] = λ
        hfile["perron_frobenius $k koopman eigenvector $index"] = w
    end
    close(hfile)
end