using StateSpacePartitions, Graphs
using KernelAbstractions
using KernelAbstractions: @kernel, @index


@info "loading data for kmeans"
hfile = h5open(data_directory  * "/ks.hdf5", "r")
timeseries = read(hfile["timeseries"])[1:2:end, 1:25:end]
close(hfile)
joined_timeseries = copy(timeseries)
i = 1
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 2
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 3
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 4
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 5
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 6
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 7
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 8
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 9
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 10
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 11
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 12
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 13
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 14
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 15
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 16
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 17
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 18
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 19
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 20
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 21
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 22
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 23
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 24
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 25
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 26
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 27
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 28
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 29
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 30
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 31
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 32
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 33
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 34
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 35
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 36
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 37
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 38
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 39
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 40
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 41
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 42
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 43
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 44
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 45
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 46
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 47
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 48
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 49
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 50
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 51
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 52
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 53
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 54
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 55
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 56
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 57
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 58
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 59
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 60
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 61
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 62
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))
i = 63
joined_timeseries = hcat(joined_timeseries, circshift(timeseries, (i, 0)))

@info "starting k-means"
p_min = 1.5e-6
@info "computing embedding"
Nmax = 50 * round(Int, 1/ p_min)
skip = 1 # maximum([round(Int, size(joined_timeseries)[2] / Nmax), 1])
tic = Base.time()
F, H, edge_information, parent_to_children, global_to_local, centers_list, CC, local_to_global = unstructured_tree(joined_timeseries[:, 1:skip:end], p_min; threshold = 1.0);
toc = Base.time() 
println("time for kmeans: ", toc - tic, " seconds")

probabilities_for_coarsening = [edge_info[3] for edge_info in edge_information]
probabilities_for_coarsening = [1.0 , probabilities_for_coarsening...]

# _, _, G, _ = graph_from_edge_information(edge_information)
# G = DiGraph(G)
# For unstructured_tree
# CC should be called centers_list
# and centers_list should be called splitting_list
# F should be called local_embedding_index_to_timeseries_index
# H should be called parent_index_to_split_timeseries_indices
# For unstructured_coarsen_edges
# graph_edges should be called edge_information
embedding = UnstructuredTree(global_to_local, centers_list, parent_to_children)

@info "loading data and computing symmetry"
hfile = h5open(data_directory * "/ks.hdf5", "r")
timeseries = read(hfile["timeseries"])[1:2:end, 1:end]
close(hfile)
joined_timeseries = copy(timeseries)

partitions = zeros(Int64, size(joined_timeseries)[2], 64)
@info "computing partition trajectory"
@kernel function compute_partition_trajectory!(embedding, partitions, joined_timeseries)
    i = @index(Global, Linear)
    @inbounds partitions[i] = embedding(joined_timeseries[:, i])
end
tic = Base.time()
for i in ProgressBar(0:63)
    compute_partition_trajectory!(KernelAbstractions.CPU(), 256, size(partitions)[1])(embedding, view(partitions,:, i+1), circshift(joined_timeseries, (i, 0)))
end
toc = Base.time()
println("time for computing partition trajectory: ", toc - tic, " seconds")

@info "saving embedding"
hfile = h5open(data_directory  * "/embedding.hdf5", "w")
hfile["markov_chain"] = partitions
hfile["probability"] = p_min
close(hfile)


@info "coarsening with different probabilities"
probabilities = [10^(-(i/4)) for i in 0:round(Int, -log(p_min)/log(10^(1/4)))]
local_to_locals = []
local_to_globals = []
for i in ProgressBar(eachindex(probabilities))
    # local_to_local, local_to_global = new_unstructured_coarsen_edges(edge_information, probabilities[i], parent_to_children, G, global_to_local)
    local_to_coarse_local, coarse_local_to_coarse_global = new_unstructured_coarsening_p2c(probabilities[i], parent_to_children, probabilities_for_coarsening, global_to_local)
    push!(local_to_locals, local_to_coarse_local)
    push!(local_to_globals, coarse_local_to_coarse_global)
end

@info "saving coarse embeddings"
hfile = h5open(data_directory  * "/embedding.hdf5", "r+")
hfile["coarse_probabilities"] = probabilities
coarse_partitions = zeros(Int64, size(joined_timeseries)[2], 64)
for j in ProgressBar(eachindex(probabilities))
    for i in ProgressBar(eachindex(partitions[:, 1]))
        for k in 1:64
            @inbounds coarse_partitions[i, k] = local_to_locals[j][partitions[i, k]]
        end
    end
    hfile["coarse_markov_chains $j"] = copy(coarse_partitions)
end
close(hfile)
##
@info "saving centers"
hfile = h5open(data_directory  * "/centers.hdf5", "w")
for i in ProgressBar(eachindex(probabilities))
    centers_list = zeros(size(timeseries)[1], length(local_to_globals[i]))
    for j in eachindex(local_to_globals[i])
        centers_list[:, j] = CC[local_to_globals[i][j]]
    end
    hfile["centers $i"] = centers_list
end
centers_list = zeros(size(timeseries)[1], length(local_to_global))
for j in eachindex(local_to_global)
    centers_list[:, j] = CC[local_to_global[j]]
end
hfile["centers"] = centers_list
close(hfile)

# sac = mean(timeseries .* circshift(timeseries, (32, 0)), dims = 2) 
# p = steady_state(partitions)
# sac_ens = sum(centers_list .* circshift(centers_list, (32, 0)) .* reshape(p, (1, 104247)), dims = 2)
# energy_timeseries = [sum(joined_timeseries[:, i] .^2) for i in 1:size(joined_timeseries)[2]]
# energy_cell_centers = [sum(centers_list[:, i] .^2) for i in 1:size(centers_list)[2]]
# p = steady_state(partitions)
# hist(energy_cell_centers, normalization = :pdf, bins = 100)
# hist!(energy_timeseries, bins = 100, normalization = :pdf)
# abs(mean(energy_timeseries) - sum(energy_cell_centers .* p))/mean(energy_timeseries)
# abs(var(energy_timeseries) - sum((energy_cell_centers .^2) .* p) + (sum(energy_cell_centers .* p) .^2))/var(energy_timeseries)

# hfile = h5open(data_directory  * "/embedding.hdf5", "r")
# markov_chain = read(hfile["coarse_markov_chains 15"])
# Q = generator(markov_chain)
# ri = decomposition(Q)
# H = ri.volume_preserving
# Λ, V = eigen(H)
# eΛ = exp.(Λ)
# hist(imag.(Λ), normalization = :pdf, bins = 100)
# hist(atan.(real.(eΛ), imag.(eΛ)), bins = 200)

# Q = sparse_generator(partitions) 
# nzs = [length(Q[:, i].nzval) for i in 1:size(Q)[1]]

##
#=
L_Q = sparse_generator(partitions[:, 1])
exits = [length(L_Q[:, i].nzval) for i in ProgressBar(1:size(L_Q)[2])]
entrances = [length(L_Q[i, :].nzval) for i in ProgressBar(1:size(L_Q)[1])]
L_Q_2 = sparse_generator(partitions[:, 16])
L = (L_Q .+ L_Q_2) / 2

function transition_counts_sparse(partitions::AbstractMatrix{<:Integer}; progress = true)
    T, K = size(partitions)
    ns = maximum(partitions)
    if progress
        inds = ProgressBar(1:K)
    else
        inds = 1:K
    end
    # Sum sparse count matrices across columns; duplicate indices are summed inside sparse(...)
    return sum(
        sparse(@view(partitions[2:T, j]),
               @view(partitions[1:T-1, j]),
               ones(Int32, T-1),  # smaller V saves memory
               ns, ns)
        for j in inds
    )
end
L_count = sparse_count_operator(partitions, maximum(partitions))

fig = Figure()
ax = Axis(fig[1, 1])
scatter!(ax, sort(entrances), color = (:blue, 0.5))
ax = Axis(fig[1, 2])
scatter!(ax, sort(exits), color = (:red, 0.5))
save("KSFigures/exits_and_entrances.png", fig)

##
tmp = [(partitions[i+1,j], partitions[i,j]) for i in 1:10000000, j in ProgressBar(1:64)]
utmp = union(tmp)
=#


