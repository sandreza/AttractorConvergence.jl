using StateSpacePartitions, Graphs

@info "loading data for kmeans"
hfile = h5open(pwd() * data_directory  * "/lorenz.hdf5", "r")
m_timeseries = read(hfile["timeseries"])
s_timeseries = read(hfile["symmetrized timeseries"])
joined_timeseries = hcat(m_timeseries, s_timeseries) # only for Partitioning Purpose
close(hfile)
@info "starting k-means"
p_min = 1e-4
@info "computing embedding"
Nmax = 100 * round(Int, 1/ p_min)
skip = maximum([round(Int, size(joined_timeseries)[2] / Nmax), 1])
F, H, edge_information, parent_to_children, global_to_local, centers_list, CC, local_to_global = unstructured_tree(joined_timeseries[:, 1:skip:end], p_min; threshold = 1.5);
_, _, G, _ = graph_from_edge_information(edge_information)
G = DiGraph(G)
# For unstructured_tree
# CC should be called centers_list
# and centers_list should be called splitting_list
# F should be called local_embedding_index_to_timeseries_index
# H should be called parent_index_to_split_timeseries_indices
# For unstructured_coarsen_edges
# graph_edges should be called edge_information
embedding = UnstructuredTree(global_to_local, centers_list, parent_to_children)

probabilities = [2^(-i) for i in 0:-log(p_min)/log(2)]

local_to_locals = []
local_to_globals = []
@info "coarsening with different probabilities"
for i in ProgressBar(eachindex(probabilities))
    local_to_local, local_to_global = new_unstructured_coarsen_edges(edge_information, probabilities[i], parent_to_children, G, global_to_local)
    push!(local_to_locals, local_to_local)
    push!(local_to_globals, local_to_global)
end

partitions = zeros(Int64, size(joined_timeseries)[2])
coarse_partitions = zeros(Int64, size(joined_timeseries)[2], length(probabilities))
@info "computing partition trajectory"
for i in ProgressBar(eachindex(partitions))
    @inbounds partitions[i] = embedding(joined_timeseries[:, i])
    for j in eachindex(probabilities)
        @inbounds coarse_partitions[i, j] = local_to_locals[j][partitions[i]]
    end
end

##
@info "saving embeddings"
hfile = h5open(pwd() * data_directory  * "/embedding.hdf5", "w")
hfile["markov_chain"] = partitions
hfile["coarse_markov_chains"] = coarse_partitions
hfile["probability"] = p_min
hfile["coarse_probabilities"] = probabilities
close(hfile)

##
@info "saving centers"
hfile = h5open(pwd() * data_directory  * "/centers.hdf5", "w")
for i in ProgressBar(eachindex(probabilities))
    centers_list = zeros(3, length(local_to_globals[i]))
    for j in eachindex(local_to_globals[i])
        centers_list[:, j] = CC[local_to_globals[i][j]]
    end
    hfile["centers $i"] = centers_list
end
centers_list = zeros(3, length(local_to_global))
for j in eachindex(local_to_global)
    centers_list[:, j] = CC[local_to_global[j]]
end
hfile["centers"] = centers_list
close(hfile)