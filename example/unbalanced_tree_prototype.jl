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
        r0 = kmeans(view(timeseries, :, indices), numstates; max_iters=10000)
        child_0 = (r0.assignments .== 1)
        ind1 = [i for (j, i) in enumerate(indices) if child_0[j] == 1]
        ind2 = [i for (j, i) in enumerate(indices) if child_0[j] == 0]
        return ind1, ind2
    end
    return [], []
end

centers, children = split(timeseries)
centers1, children2 = split(children[1])
ind1, ind2 = split(timeseries, 1:size(timeseries)[2], 10^8)
ind3, ind4 = split(timeseries, ind1)
issetequal(setdiff(ind1, ind3), ind4)
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
            push!(G, qOld)
        end
    end
    return F, G, H, P2
end

##
F, G, H, PI = unstructured_tree(timeseries, 0.1)
node_labels, adj, adj_mod, edge_numbers = graph_from_PI(PI)
nn = maximum([PI[i][2] for i in eachindex(PI)])
node_labels = ones(nn)
[node_labels[PI[i][2]] = PI[i][3] for i in eachindex(PI)]
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