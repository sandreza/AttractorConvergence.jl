using Statistics, HDF5, LinearAlgebra

hfile = h5open(data_directory  * "/structured_embedding_revision.hdf5", "r+")
kmeans_distances = read(hfile["kmeans_distances"])
close(hfile)

min_distances = [minimum(dist) for dist in eachcol(kmeans_distances)]
sigma = maximum(min_distances)
sigma_2 = sigma/10
sigma_3 = mean(min_distances)

radial_basis_functions = exp.(-kmeans_distances.^2 / (2 * sigma^2))
normalized_radial_basis_functions = radial_basis_functions ./ sum(radial_basis_functions, dims=1)

radial_basis_functions_2 = exp.(-kmeans_distances.^2 / (2 * (sigma_2)^2))
normalized_radial_basis_functions_2 = radial_basis_functions_2 ./ sum(radial_basis_functions_2, dims=1)

radial_basis_functions_3 = exp.(-kmeans_distances.^2 / (2 * (sigma_3)^2))
normalized_radial_basis_functions_3 = radial_basis_functions_3 ./ sum(radial_basis_functions_3, dims=1)

hfile = h5open(data_directory  * "/structured_embedding_revision.hdf5", "r+")
hfile["radial_basis_functions"] = radial_basis_functions
hfile["normalized_radial_basis_functions"] = normalized_radial_basis_functions
hfile["radial_basis_functions_2"] = radial_basis_functions_2
hfile["normalized_radial_basis_functions_2"] = normalized_radial_basis_functions_2
close(hfile)

hfile = h5open(data_directory  * "/structured_embedding_revision.hdf5", "r+")
hfile["radial_basis_functions_3"] = radial_basis_functions_3
hfile["normalized_radial_basis_functions_3"] = normalized_radial_basis_functions_3
close(hfile)



tic = Base.time()
rbf_operator = normalized_radial_basis_functions[:, 2:1:end] * pinv(normalized_radial_basis_functions[:, 1:1:end-1])
toc = Base.time()
@info "time for operator: ", toc - tic, " seconds"



Λ_rbf, W_rbf = eigen(rbf_operator')
Λ_rbf, V_rbf = eigen(rbf_operator)
p_rbf = real.(V_rbf[:, end])
p_rbf = p_rbf ./ sum(p_rbf)
s_rbf = scaled_entropy(p_rbf)
pratio_rbf = maximum(p_rbf) / minimum(p_rbf)

tic_2 = Base.time()
rbf_operator_2 = normalized_radial_basis_functions_2[:, 2:1:end] * pinv(normalized_radial_basis_functions_2[:, 1:1:end-1])
toc_2 = Base.time()
@info "time for operator: ", toc_2 - tic_2, " seconds"

Λ_rbf_2, W_rbf_2 = eigen(rbf_operator_2')
Λ_rbf_2, V_rbf_2 = eigen(rbf_operator_2)
p_rbf_2 = real.(V_rbf_2[:, end])
p_rbf_2 = p_rbf_2 ./ sum(p_rbf_2)
s_rbf_2 = scaled_entropy(p_rbf_2)
pratio_rbf_2 = maximum(p_rbf_2) / minimum(p_rbf_2)

hfile = h5open(data_directory  * "/structured_eigenvalues_and_generators.hdf5", "r+")
hfile["rpf_operator"] = rbf_operator
hfile["Λ_rbf"] = Λ_rbf
hfile["W_rbf"] = W_rbf
hfile["V_rbf"] = V_rbf
hfile["p_rbf"] = p_rbf
hfile["s_rbf"] = s_rbf
hfile["pratio_rbf"] = pratio_rbf
hfile["rbf_operator_time"] = toc - tic
hfile["rpf_operator_2"] = rbf_operator_2
hfile["Λ_rbf_2"] = Λ_rbf_2
hfile["W_rbf_2"] = W_rbf_2
hfile["V_rbf_2"] = V_rbf_2
hfile["p_rbf_2"] = p_rbf_2
hfile["s_rbf_2"] = s_rbf_2
hfile["pratio_rbf_2"] = pratio_rbf_2
hfile["rbf_operator_time_2"] = toc_2 - tic_2
close(hfile)

rbf_operator_3 = normalized_radial_basis_functions_3[:, 2:1:end] * pinv(normalized_radial_basis_functions_3[:, 1:1:end-1])
Λ_rbf_3, W_rbf_3 = eigen(rbf_operator_3')
Λ_rbf_3, V_rbf_3 = eigen(rbf_operator_3)
p_rbf_3 = real.(V_rbf_3[:, end])
p_rbf_3 = p_rbf_3 ./ sum(p_rbf_3)
s_rbf_3 = scaled_entropy(p_rbf_3)
pratio_rbf_3 = maximum(p_rbf_3) / minimum(p_rbf_3)

hfile = h5open(data_directory  * "/structured_eigenvalues_and_generators.hdf5", "r+")
hfile["rpf_operator_3"] = rbf_operator_3
hfile["Λ_rbf_3"] = Λ_rbf_3
hfile["W_rbf_3"] = W_rbf_3
hfile["V_rbf_3"] = V_rbf_3
hfile["p_rbf_3"] = p_rbf_3
hfile["s_rbf_3"] = s_rbf_3
hfile["pratio_rbf_3"] = pratio_rbf_3
close(hfile)


@info "loading data"
hfile = h5open(data_directory  * "/structured_embedding_revision.hdf5", "r")
kmeans_chain = read(hfile["markov_chain_kmeans"])
close(hfile)

@info "computing eigenvalues and generators"
tic = Base.time()
P_kmeans = perron_frobenius(kmeans_chain; step = 1)
toc = Base.time()
println("time for perron-frobenius: ", toc - tic, " seconds")


hfile = h5open(data_directory  * "/embedding_revision.hdf5", "r")
coarse_markov_chains = read(hfile["coarse_markov_chains 17"])
close(hfile)

tic2 = Base.time()
pf = perron_frobenius(coarse_markov_chains; step = 1)
toc2 = Base.time()
println("time for perron-frobenius: ", toc - tic, " seconds")

Λ_bskmeans = eigvals(pf)

hfile = h5open(data_directory  * "/structured_eigenvalues_and_generators.hdf5", "r+")
hfile["Λ_bskmeans"] = Λ_bskmeans
hfile["pf_bskmeans_time"] = toc2 - tic2
hfile["pf_kmeans_time"] = toc - tic
close(hfile)


