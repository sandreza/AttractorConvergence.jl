using HDF5, CairoMakie
using StateSpacePartitions, Graphs, GraphMakie
using MarkovChainHammer, ProgressBars, LinearAlgebra
using SparseArrays, NetworkLayout, Printf, Random

@info "loading data for kmeans"
hfile = h5open(data_directory  * "/lorenz.hdf5", "r")
joined_timeseries = read(hfile["timeseries"])
close(hfile)
@info "starting k-means"
p_min = 0.1
@info "computing embedding"
Nmax = 50 * round(Int, 1/ p_min)
skipind = maximum([round(Int, size(joined_timeseries)[2] / Nmax), 1])
reduced_timeseries = joined_timeseries[:, 1:skipind:end]
tic = Base.time()
Random.seed!(12345)
F, H, edge_information, parent_to_children, global_to_local, centers_list, CC, local_to_global = unstructured_tree(reduced_timeseries, p_min; threshold = 1.0);
PI = edge_information

adj_array = []
adj_mod_array = []
edge_numbers_array = []
node_labels_array = []
pmins = [0.5, 0.3, 0.2]
for p_min in pmins
    Random.seed!(12345)
    F, H, edge_information, parent_to_children, global_to_local, centers_list, CC, local_to_global = unstructured_tree(reduced_timeseries, p_min; threshold = 1.0);
    PI = edge_information

    N = maximum([maximum([PI[i][1], PI[i][2]]) for i in eachindex(PI)])
    adj = spzeros(Int64, N, N)
    adj_mod = spzeros(Float64, N, N)
    for i in ProgressBar(eachindex(PI))
        ii = PI[i][1]
        jj = PI[i][2]
        modularity_value = PI[i][3]
        adj[ii, jj] += 1
        adj_mod[ii, jj] = modularity_value
    end 
    N = maximum([maximum([PI[i][1], PI[i][2]]) for i in eachindex(PI)])
    nodel_labels = zeros(N)
    nodel_labels[1] = 1.0
    for i in eachindex(PI)
        nodel_labels[PI[i][2]] = PI[i][3] # change from PI[i][1] for root, PI[i][2] for leaf
    end
    push!(node_labels_array, nodel_labels)
    push!(adj_array, adj)
    push!(adj_mod_array, adj_mod)
    push!(edge_numbers_array, length(PI))
end


fig = Figure(resolution = (2000, 700))
layout = Buchheim()
colormap = :glasbey_hv_n256
set_theme!(backgroundcolor=:white)
for t in [1, 2, 3]
    ax11 = Axis(fig[1,t]; title =  @sprintf("threshold = %5.2f", pmins[t]), titlesize = 40)
    G = SimpleDiGraph(adj_array[t])
    transparancy = 0.4 * adj_mod_array[t].nzval[:] / adj_mod_array[t].nzval[1] .+ 0.1
    nlabels_fontsize = 30
    edge_color = [(:red, transparancy[i]) for i in 1:edge_numbers_array[t]]
    nlabels = [@sprintf("%2.2f", node_labels_array[t][i]) for i in 1:nv(G)]
    graphplot!(ax11, G, layout=layout, nlabels=nlabels, node_size=80,
        node_color=(:orange, 0.9), edge_color=edge_color, edge_width=5,
        arrow_size=45, nlabels_align=(:center, :center),
        nlabels_textsize=nlabels_fontsize, tangents=((0, -1), (0, -1)))
    hidedecorations!(ax11)
    hidespines!(ax11);
end
# save(pwd() * "/unstructured_figures" * "/Figure0.png", fig)