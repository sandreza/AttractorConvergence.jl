using HDF5
using ParallelKMeans, NetworkLayout, Graphs, Printf, GraphMakie
using HDF5, GraphMakie, NetworkLayout, MarkovChainHammer, ProgressBars, GLMakie, Graphs
using Printf, Random

include(pwd() * "/generate/data.jl")
include(pwd() * "/example/unbalanced_tree.jl")
hfile = h5open(pwd() * "/data/lorenz.hdf5", "r")
timeseries = read(hfile["timeseries"])
close(hfile)

function split(timeseries, indices, n_min)
    numstates = 2
    if length(indices) > n_min
        r0 = kmeans(view(timeseries, :, indices), numstates; max_iters=10^4)
        child_0 = (r0.assignments .== 1)
        ind1 = [i for (j, i) in enumerate(indices) if child_0[j] == 1]
        ind2 = [i for (j, i) in enumerate(indices) if child_0[j] == 0]
        return ind1, ind2
    end
    return [], []
end

function unstructured_tree(timeseries, p_min)
    n = size(timeseries)[2]
    n_min = floor(Int, p_min * n)
    W, F, G, P1, P2 = [collect(1:n)], [], [], [1], []
    H = []
    global_index = 1
    while (length(W) > 0)
        w = popfirst!(W)
        p1 = popfirst!(P1)
        ind1, ind2 = split(timeseries, w, 2 * n_min)
        if (length(ind1) > 0) & (length(ind2) > 0)
            W = [ind1, ind2, W...]
            P1 = [global_index + 1, global_index + 2, P1...]
            P2 = push!(P2, (p1, global_index + 1, length(ind1) / n))
            P2 = push!(P2, (p1, global_index + 2, length(ind2) / n))
            global_index += 2
            push!(H, [ind1, ind2])
        else
            push!(F, w)
        end
    end
    return F, G, H, P2
end

function split2(timeseries, indices, n_min; numstates = rand([2 3]))
    if length(indices) > n_min
        r0 = kmeans(view(timeseries, :, indices), numstates; max_iters=10^4)
        inds = [[i for (j, i) in enumerate(indices) if r0.assignments[j] == k] for k in 1:numstates]
        centers = [r0.centers[:, k] for k in 1:numstates]
        return inds, centers
    end
    return [[]], [[]]
end

function unstructured_tree2(timeseries, p_min)
    n = size(timeseries)[2]
    n_min = floor(Int, p_min * n)
    W, F, G, P1, P2 = [collect(1:n)], [], [], [1], []
    H = []
    global_index = 1
    while (length(W) > 0)
        w = popfirst!(W)
        p1 = popfirst!(P1)
        inds, centers = split2(timeseries, w, 2 * n_min)
        if all([length(ind) > 0 for ind in inds])
            W = [inds..., W...]
            Ptmp = []
            [push!(Ptmp, global_index + i) for i in eachindex(inds)]
            P1 = [Ptmp..., P1...]
            [push!(P2, (p1, global_index + i, length(ind) / n)) for (i, ind) in enumerate(inds)]
            global_index += length(inds)
            push!(H, [inds...])
        else
            push!(F, w)
        end
    end
    return F, G, H, P2
end

function children_from_PI(PI, parent_index)
    return [PI[i][2] for i in eachindex(PI) if PI[i][1] == parent_index]
end

function leaf_nodes_from_PI(PI)
    return [PI[i][2] for i in eachindex(PI) if length(children_from_PI(PI, PI[i][2])) == 0]
end

##
F, G, H, PI = unstructured_tree2(timeseries, 0.0002)
node_labels, adj, adj_mod, edge_numbers = graph_from_PI(PI)
nn = maximum([PI[i][2] for i in eachindex(PI)])
node_labels = ones(nn)
probabilities = [node_labels[PI[i][2]] = PI[i][3] for i in eachindex(PI)]
node_labels = collect(1:nn)
leaf_inds = leaf_nodes_from_PI(PI)
scaled_entropy([probabilities[leaf-1] for leaf in leaf_inds])
##
fig = Figure(resolution=(2 * 800, 800))
layout = Buchheim()
colormap = :glasbey_hv_n256
set_theme!(backgroundcolor=:white)

ax11 = Axis(fig[1, 1])
G = SimpleDiGraph(adj)
transparancy = 0.4 * adj_mod.nzval[:] / adj_mod.nzval[1] .+ 0.1
nlabels_fontsize = 35
edge_color = [(:red, transparancy[i]) for i in 1:edge_numbers]
nlabels = [@sprintf("%2.2f", node_labels[i]) for i in 1:nv(G)]
graphplot!(ax11, G, layout=layout, nlabels=nlabels, node_size=100,
    node_color=(:orange, 0.9), edge_color=edge_color, edge_width=5,
    arrow_size=45, nlabels_align=(:center, :center),
    nlabels_textsize=nlabels_fontsize, tangents=((0, -1), (0, -1)))
# cc = cameracontrols(ax11.scene)
hidedecorations!(ax11)
hidespines!(ax11);
display(fig)