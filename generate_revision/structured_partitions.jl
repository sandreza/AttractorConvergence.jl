using StateSpacePartitions, Graphs
using KernelAbstractions
using KernelAbstractions: @kernel, @index

@info "loading data for kmeans"
hfile = h5open(data_directory  * "/lorenz_revision.hdf5", "r")
m_timeseries = read(hfile["timeseries"])
dt = read(hfile["dt"])
close(hfile)

@info "Ulam's method"
xmin, xmax = extrema(m_timeseries[1, :])
ymin, ymax = extrema(m_timeseries[2, :])
zmin, zmax = extrema(m_timeseries[3, :])
##
cells = 10 # yields 250 states, roughtly, 2.5 * cells^2
xgrid = collect(range(xmin, stop=xmax, length=cells+2))[2:end-1]
ygrid = collect(range(ymin, stop=ymax, length=cells+2))[2:end-1]
zgrid = collect(range(zmin, stop=zmax, length=cells+2))[2:end-1]

tuple_list = Tuple{Int64, Int64, Int64}[]
tic = Base.time()
@info "tensor product grid"
for i in ProgressBar(1:size(m_timeseries,2))
    s⃗ = m_timeseries[:, i]
    ind1 = sum(s⃗[1] .< xgrid)
    ind2 = sum(s⃗[2] .< ygrid)
    ind3 = sum(s⃗[3] .< zgrid)
    push!(tuple_list, Tuple([ind1, ind2, ind3]))
end
@info "redundant states removed"
markov_state = union(tuple_list)
##
markov_index = Int64[]
for i in ProgressBar(eachindex(tuple_list))
    push!(markov_index, argmax([tuple_list[i]] .== markov_state))
end
toc = Base.time()
@info "time for ulam: ", toc - tic, " seconds"


@info "saving"
hfile = h5open(data_directory  * "/structured_embedding_revision.hdf5", "w")
hfile["markov_chain_ulam"] = markov_index
hfile["dt"] = dt
hfile["ulam_time"] = toc - tic
close(hfile)

@info "kmeans"
tic = Base.time()
r0 = kmeans(m_timeseries[:, 1:100:end],  length(markov_state); max_iters=10000)
toc = Base.time()
@info "time for kmeans: ", toc - tic, " seconds"

tic = Base.time()
kmeans_index = zeros(Int, size(m_timeseries, 2))
distances = zeros(length(markov_state), size(m_timeseries, 2))
for i in ProgressBar(eachindex(kmeans_index))
    @inbounds state = m_timeseries[:, i]
    tmp = [norm(state - center) for center in eachcol(r0.centers)]
    @inbounds kmeans_index[i] = argmin(tmp)
    @inbounds distances[:, i] = tmp
end
toc = Base.time()
@info "time for kmeans index: ", toc - tic, " seconds"

hfile = h5open(data_directory  * "/structured_embedding_revision.hdf5", "r+")
hfile["markov_chain_kmeans"] = kmeans_index
hfile["kmeans_distances"] = distances
hfile["kmeans_centers"] = r0.centers
hfile["kmeans_time"] = toc - tic
close(hfile)

##

@info "starting bisecting k-means"
p_min = 1.4 * 1 / length(markov_state)
@info "computing embedding"
Nmax = 50 * round(Int, 1/ p_min)
skip_index = maximum([round(Int, size(m_timeseries)[2] / Nmax), 1])
tic = Base.time()
F, H, edge_information, parent_to_children, global_to_local, centers_list, CC, local_to_global = unstructured_tree(m_timeseries[:, 1:skip_index:end], p_min; threshold = 1.0);
toc2 = Base.time()
println("time for kmeans: ", toc2 - tic, " seconds")
# graph_edges should be called edge_information
embedding = UnstructuredTree(global_to_local, centers_list, parent_to_children)
partitions = zeros(Int64, size(m_timeseries)[2])
@info "computing partition trajectory"
tic2 = Base.time()
for i in ProgressBar(eachindex(partitions))
    @inbounds partitions[i] = embedding(m_timeseries[:, i])
end
toc = Base.time()
println("time for computing partition trajectory: ", toc - tic2, " seconds")

centers_list = zeros(3, length(local_to_global))
for j in eachindex(local_to_global)
    centers_list[:, j] = CC[local_to_global[j]]
end

hfile = h5open(data_directory  * "/structured_embedding_revision.hdf5", "r+")
hfile["markov_chain_unstructured"] = partitions
hfile["unstructured_centers"] = centers_list
hfile["unstructured_time"] = toc - tic2
close(hfile)