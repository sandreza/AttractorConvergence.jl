using HDF5, GraphMakie, NetworkLayout, MarkovChainHammer, ProgressBars, CairoMakie, Graphs
using Printf, Random
Random.seed!(12345)

include(pwd() * "/generate/data.jl")
hfile = h5open(pwd() * "/data/embedding.hdf5", "r")
markov_chain = read(hfile["markov_chain"])
markov_chain_reduced = div.(markov_chain .- 1, Ref(32)) .+ 1 
close(hfile)

include("unbalanced_tree.jl")

P = perron_frobenius(markov_chain_reduced; step = 10)
Q = generator(markov_chain_reduced)
F, G, H, PI = leicht_newman_with_tree(P, 0.0)
node_labels, adj, adj_mod, edge_numbers = graph_from_PI(PI)
node_labels = Int.(collect(eachindex(node_labels)))
##
fig = Figure(resolution = (2*800, 800))
layout = Buchheim()
colormap = :glasbey_hv_n256
set_theme!(backgroundcolor=:white)

ax11 = Axis(fig[1,1])
G = SimpleDiGraph(adj)
transparancy = 0.4 * adj_mod.nzval[:] / adj_mod.nzval[1] .+ 0.1
nlabels_fontsize = 35
edge_color = [(:red, transparancy[i]) for i in 1:edge_numbers]
nlabels = [@sprintf("%2.2f", node_labels[i]) for i in 1:nv(G)]
graphplot!(ax11, G, layout=layout, nlabels=nlabels, node_size=100,
    node_color=(:orange, 0.9), edge_color=edge_color, edge_width=5,
    arrow_size=45, nlabels_align=(:center, :center),
    nlabels_textsize=nlabels_fontsize, tangents=((0, -1), (0, -1)))
ax12 = LScene(fig[1,2]; show_axis = false)
layout = Spectral(dim=3)
[Q[i, i] = 0 for i in 1:size(Q)[1]]
g = DiGraph(Q)
graphplot!(ax12, g, layout=layout, node_size=0.0, edge_width=1.0)
# cc = cameracontrols(ax11.scene)
hidedecorations!(ax11)
hidespines!(ax11);
display(fig)