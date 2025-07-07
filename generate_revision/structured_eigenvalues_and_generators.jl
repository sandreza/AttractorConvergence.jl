using HDF5, MarkovChainHammer, ProgressBars, LinearAlgebra, Statistics, Random, SparseArrays
using StateSpacePartitions

@info "loading data"
hfile = h5open(data_directory  * "/structured_embedding_revision.hdf5", "r")
unstructured_chain = read(hfile["markov_chain_unstructured"])
kmeans_chain = read(hfile["markov_chain_kmeans"])
ulam_chain = read(hfile["markov_chain_ulam"])
close(hfile)

@info "computing eigenvalues and generators"
tic = Base.time()
P_ulam = perron_frobenius(ulam_chain; step = 1)
toc = Base.time()
Λ_ulam, W_ulam = eigen(P_ulam')
Λ_ulam, V_ulam = eigen(P_ulam)
p_ulam = real.(V_ulam[:, end])
p_ulam = p_ulam ./ sum(p_ulam)
s_ulam = scaled_entropy(p_ulam)
pratio_ulam = maximum(p_ulam) / minimum(p_ulam)

P_kmeans = perron_frobenius(kmeans_chain; step = 1)
Λ_kmeans, W_kmeans = eigen(P_kmeans')
Λ_kmeans, V_kmeans = eigen(P_kmeans)
p_kmeans = real.(V_kmeans[:, end])
p_kmeans = p_kmeans ./ sum(p_kmeans)
s_kmeans = scaled_entropy(p_kmeans)
pratio_kmeans = maximum(p_kmeans) / minimum(p_kmeans)

P_unstructured = perron_frobenius(unstructured_chain; step = 1)
Λ_unstructured, W_unstructured = eigen(P_unstructured')
Λ_unstructured, V_unstructured = eigen(P_unstructured)
p_unstructured = real.(V_unstructured[:, end])
p_unstructured = p_unstructured ./ sum(p_unstructured)
s_unstructured = scaled_entropy(p_unstructured)
pratio_unstructured = maximum(p_unstructured) / minimum(p_unstructured)

@info "saving data"
hfile = h5open(data_directory  * "/structured_eigenvalues_and_generators.hdf5", "w")
hfile["p_ulam"] = p_ulam
hfile["p_kmeans"] = p_kmeans
hfile["p_unstructured"] = p_unstructured
hfile["s_ulam"] = s_ulam
hfile["s_kmeans"] = s_kmeans
hfile["s_unstructured"] = s_unstructured
hfile["pratio_ulam"] = pratio_ulam
hfile["pratio_kmeans"] = pratio_kmeans
hfile["pratio_unstructured"] = pratio_unstructured
hfile["Λ_ulam"] = Λ_ulam
hfile["Λ_kmeans"] = Λ_kmeans
hfile["Λ_unstructured"] = Λ_unstructured
hfile["W_ulam"] = W_ulam
hfile["W_kmeans"] = W_kmeans
hfile["W_unstructured"] = W_unstructured
hfile["V_ulam"] = V_ulam
hfile["V_kmeans"] = V_kmeans
hfile["V_unstructured"] = V_unstructured
close(hfile)

# larger step
@info "computing eigenvalues and generators for larger step"
P_ulam = perron_frobenius(ulam_chain; step = 2)
P_ulam10 = copy(P_ulam)
Λ_ulam, W_ulam = eigen(P_ulam')
Λ_ulam, V_ulam = eigen(P_ulam)
p_ulam = real.(V_ulam[:, end])
p_ulam = p_ulam ./ sum(p_ulam)
s_ulam = scaled_entropy(p_ulam)
pratio_ulam = maximum(p_ulam) / minimum(p_ulam)

P_kmeans = perron_frobenius(kmeans_chain; step = 2)
Λ_kmeans, W_kmeans = eigen(P_kmeans')
Λ_kmeans, V_kmeans = eigen(P_kmeans)
p_kmeans = real.(V_kmeans[:, end])
p_kmeans = p_kmeans ./ sum(p_kmeans)
s_kmeans = scaled_entropy(p_kmeans)
pratio_kmeans = maximum(p_kmeans) / minimum(p_kmeans)

P_unstructured = perron_frobenius(unstructured_chain; step = 2)
Λ_unstructured, W_unstructured = eigen(P_unstructured')
Λ_unstructured, V_unstructured = eigen(P_unstructured)
p_unstructured = real.(V_unstructured[:, end])
p_unstructured = p_unstructured ./ sum(p_unstructured)
s_unstructured = scaled_entropy(p_unstructured)
pratio_unstructured = maximum(p_unstructured) / minimum(p_unstructured)

@info "saving data for larger step"
hfile = h5open(data_directory  * "/structured_eigenvalues_and_generators_coarse.hdf5", "w")
hfile["p_ulam"] = p_ulam
hfile["p_kmeans"] = p_kmeans
hfile["p_unstructured"] = p_unstructured
hfile["s_ulam"] = s_ulam
hfile["s_kmeans"] = s_kmeans
hfile["s_unstructured"] = s_unstructured
hfile["pratio_ulam"] = pratio_ulam
hfile["pratio_kmeans"] = pratio_kmeans
hfile["pratio_unstructured"] = pratio_unstructured
hfile["Λ_ulam"] = Λ_ulam
hfile["Λ_kmeans"] = Λ_kmeans
hfile["Λ_unstructured"] = Λ_unstructured
hfile["W_ulam"] = W_ulam
hfile["W_kmeans"] = W_kmeans
hfile["W_unstructured"] = W_unstructured
hfile["V_ulam"] = V_ulam
hfile["V_kmeans"] = V_kmeans
hfile["V_unstructured"] = V_unstructured
close(hfile)


# large step but different data | NOT APPLES TO APPLES | DIFFERENT EMBEDDING FOR KMEANS
@info "computing eigenvalues and generators for large step but different data"
hfile = h5open(data_directory  * "/structured_embedding_revision_coarse.hdf5", "r")
unstructured_chain = read(hfile["markov_chain_unstructured"])
kmeans_chain = read(hfile["markov_chain_kmeans"])
ulam_chain = read(hfile["markov_chain_ulam"])
close(hfile)

P_ulam = perron_frobenius(ulam_chain; step = 1)
Λ_ulam, W_ulam = eigen(P_ulam')
Λ_ulam, V_ulam = eigen(P_ulam)
p_ulam = real.(V_ulam[:, end])
p_ulam = p_ulam ./ sum(p_ulam)
s_ulam = scaled_entropy(p_ulam)
pratio_ulam = maximum(p_ulam) / minimum(p_ulam)

P_kmeans = perron_frobenius(kmeans_chain; step = 1)
Λ_kmeans, W_kmeans = eigen(P_kmeans')
Λ_kmeans, V_kmeans = eigen(P_kmeans)
p_kmeans = real.(V_kmeans[:, end])
p_kmeans = p_kmeans ./ sum(p_kmeans)
s_kmeans = scaled_entropy(p_kmeans)
pratio_kmeans = maximum(p_kmeans) / minimum(p_kmeans)

P_unstructured = perron_frobenius(unstructured_chain; step = 1)
Λ_unstructured, W_unstructured = eigen(P_unstructured')
Λ_unstructured, V_unstructured = eigen(P_unstructured)
p_unstructured = real.(V_unstructured[:, end])
p_unstructured = p_unstructured ./ sum(p_unstructured)
s_unstructured = scaled_entropy(p_unstructured)
pratio_unstructured = maximum(p_unstructured) / minimum(p_unstructured)

@info "saving data for large step but different data"
hfile = h5open(data_directory  * "/structured_eigenvalues_and_generators_coarse_different.hdf5", "w")
hfile["p_ulam"] = p_ulam
hfile["p_kmeans"] = p_kmeans
hfile["p_unstructured"] = p_unstructured
hfile["s_ulam"] = s_ulam
hfile["s_kmeans"] = s_kmeans
hfile["s_unstructured"] = s_unstructured
hfile["pratio_ulam"] = pratio_ulam
hfile["pratio_kmeans"] = pratio_kmeans
hfile["pratio_unstructured"] = pratio_unstructured
hfile["Λ_ulam"] = Λ_ulam
hfile["Λ_kmeans"] = Λ_kmeans
hfile["Λ_unstructured"] = Λ_unstructured
hfile["W_ulam"] = W_ulam
hfile["W_kmeans"] = W_kmeans
hfile["W_unstructured"] = W_unstructured
hfile["V_ulam"] = V_ulam
hfile["V_kmeans"] = V_kmeans
hfile["V_unstructured"] = V_unstructured
close(hfile)
