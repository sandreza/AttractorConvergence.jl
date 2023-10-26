using HDF5
using ParallelKMeans, NetworkLayout, Graphs, Printf, GraphMakie
using HDF5, GraphMakie, NetworkLayout, MarkovChainHammer, ProgressBars, GLMakie, Graphs
using Printf, Random, SparseArrays

Random.seed!(12345)

include(pwd() * "/generate/data.jl")
include(pwd() * "/example/unbalanced_tree.jl")
hfile = h5open(pwd() * "/data/lorenz.hdf5", "r")
timeseries = read(hfile["timeseries"])
s_timeseries = read(hfile["symmetrized timeseries"])
dt= read(hfile["dt"])
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

function split2(timeseries, indices, n_min; numstates = 2)
    if length(indices) > n_min
        r0 = kmeans(view(timeseries, :, indices), numstates; max_iters=10^4)
        inds = [[i for (j, i) in enumerate(indices) if r0.assignments[j] == k] for k in 1:numstates]
        centers = [r0.centers[:, k] for k in 1:numstates]
        return inds, centers
    end
    return [[]], [[]]
end

function unstructured_tree2(timeseries, p_min; threshold = 2)
    n = size(timeseries)[2]
    n_min = floor(Int, threshold * p_min * n)
    W, F, G, P1, P2 = [collect(1:n)], [], [], [1], []
    H = []
    C = Dict()
    P3 = Dict()
    P4 = Dict()
    P5 = Dict()
    CC = Dict()
    leaf_index = 1
    global_index = 1
    while (length(W) > 0)
        w = popfirst!(W)
        p1 = popfirst!(P1)
        inds, centers = split2(timeseries, w, n_min)
        C[p1] =  centers
        if all([length(ind) > 0 for ind in inds])
            W = [inds..., W...]
            Ptmp = []
            [push!(Ptmp, global_index + i) for i in eachindex(inds)]
            P1 = [Ptmp..., P1...]
            [push!(P2, (p1, global_index + i, length(ind) / n)) for (i, ind) in enumerate(inds)]
            Ptmp2 = Int64[]
            [push!(Ptmp2, global_index + i) for (i, ind) in enumerate(inds)]
            [CC[global_index + i] = centers[i] for i in eachindex(inds)]
            P3[p1] = Ptmp2
            global_index += length(inds)
            push!(H, [inds...])
        else
            push!(F, w)
            push!(H, [[]])
            P3[p1] = NaN
            P4[p1] = leaf_index
            P5[leaf_index] = p1
            leaf_index += 1
        end
    end
    return F, G, H, P2, P3, P4, C, CC, P5
end

function children_from_PI(PI, parent_index)
    return [PI[i][2] for i in eachindex(PI) if PI[i][1] == parent_index]
end
function parent_to_children_map(PI)
end
# slow version
function leaf_nodes_from_PI(PI)
    #=
    leafs = Int64[]
    for i in ProgressBar(eachindex(PI))
        if length(children_from_PI(PI, PI[i][2])) == 0
            push!(leafs, PI[i][2])
        end
    end
    return leafs
    =#
    return [PI[i][2] for i in eachindex(PI) if length(children_from_PI(PI, PI[i][2])) == 0]
end
# To do list 
# given parent index, output children 
# given leaf node global index, output local index 
# write embedding function 
# compare with power_tree.jl
##
@info "constructing data"
kmeans_data = timeseries# hcat(timeseries[:, 1:100:end], s_timeseries[:, 1:100:end])
@info "constructing tree"
F, G, H, PI, P3, P4, C, CC, P5 = unstructured_tree2(kmeans_data, 0.000175);
@info "getting node labels"
node_labels, adj, adj_mod, edge_numbers = graph_from_PI(PI);
nn = maximum([PI[i][2] for i in eachindex(PI)]);
node_labels = ones(nn)
probabilities = [node_labels[PI[i][2]] = PI[i][3] for i in eachindex(PI)];
probabilities = vcat([1], probabilities)
node_labels = probabilities
node_labels = collect(1:nn)
n = size(kmeans_data)[2]
c =  [length(f)/n for f in F]
leaf_probabilities = c
se = scaled_entropy(leaf_probabilities)
pr = maximum(leaf_probabilities) / minimum(leaf_probabilities)
println("scaled entropy $se and ratio $pr")

struct UnstructuredTree{L, C, CH}
    leafmap::L 
    centers::C 
    children::CH
end

function (embedding::UnstructuredTree)(state)
    current_index = 1
    while length(embedding.centers[current_index]) > 1
        local_child = argmin([norm(state - center) for center in embedding.centers[current_index]])
        current_index = embedding.children[current_index][local_child]
    end
    return embedding.leafmap[current_index]
end
##
embedding = UnstructuredTree(P4, C, P3)
##
me = Int64[]
@info "computing markov embedding"
for state in ProgressBar(eachcol(timeseries))
    push!(me, embedding(state))
end
@info "computing symmetric embedding"
me_s = Int64[]
for state in ProgressBar(eachcol(s_timeseries))
    push!(me_s, embedding(state))
end
##
@info "constructing generator and computing steady states"
Q = generator(me; dt = dt)
Qs = generator(me_s; dt = dt)
Q = (Q + Qs) /2
Λ, V =  eigen(Q)
p = steady_state(Q)
##
@info "Viz eigenvalues"
hfile = h5open(pwd() * "/data/embedding.hdf5", "r")
hfile2 = h5open(pwd() * "/data/kmeans.hdf5", "r")
centers_matrix = read(hfile2["centers"])
centers_list = [[centers_matrix[:, 1, i], centers_matrix[:, 2, i]] for i in 1:size(centers_matrix)[3]]
centers = get_markov_states(centers_list, 12)
markov_chain = read(hfile["markov_chain"])
s_markov_chain = read(hfile["symmetrized markov chain"])
dt2 = read(hfile["dt"])
close(hfile2)
close(hfile)
Q2 = (generator(markov_chain; dt = dt2) + generator(s_markov_chain; dt = dt2))/2
Λ2, V2 =  eigen(Q2)
p2 = steady_state(Q2)
##
fig = Figure()
ax11 = Axis(fig[1,1]; title = "uniform probability")
xlower = 1000
xupper = 100
yupper = xlower/ 2
scatter!(ax11, real.(Λ), imag.(Λ), color = (:red, 0.1))
xlims!(ax11, -xlower, xupper)
ylims!(ax11, -yupper, yupper)
ax12 = Axis(fig[1,2]; title = "power tree")
scatter!(ax12, real.(Λ2), imag.(Λ2), color = (:blue, 0.1))
xlims!(ax12, -xlower, xupper)
ylims!(ax12, -yupper, yupper)
display(fig)
ax21 = Axis(fig[2,1]; title = "uniform probability")
scatter!(ax21, sort(real.(p)), color = (:red, 0.1))
ylims!(ax21, 0, 0.001)
ax22 = Axis(fig[2,2]; title = "power tree")
scatter!(ax22, sort(real.(p2)), color = (:blue, 0.1))
ylims!(ax22, 0, 0.001)
display(fig)
##
if nn < 30
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
end

##
W = koopman_modes(Q)
step = 1
Pmat = (perron_frobenius(me; step = step) + perron_frobenius(me_s; step = step))/2 # matrix exponential also works
ΛP, VP = eigen(Pmat)
W3 = inv(VP)
W2 = koopman_modes(Q2)
##
# 3, 8, 13
koopman_mode = real.(W3[end-3, :])
inds = 1:10:size(timeseries)[2]
koopman_mode = koopman_mode ./ sign(koopman_mode[end])
colors = [koopman_mode[me[j]] for j in inds]
colormap = :balance # :plasma # :glasbey_hv_n256 # :balance
q = 0.05
blue_quant = quantile(colors, q)
red_quant = quantile(colors, 1-q)
##
set_theme!(backgroundcolor=:black)
fig = Figure()
ax = LScene(fig[1,1]; show_axis = false)
markersize = 4
scatter!(ax, timeseries[:, inds]; color=colors, colormap=colormap, markersize=markersize, colorrange = (blue_quant, red_quant))
display(fig)
##
set_theme!(backgroundcolor=:white)
layout = Spectral(dim=3)
tmpQ = copy(Q)
[tmpQ[i, i] = 0 for i in 1:size(Q)[1]]
g = DiGraph(tmpQ)
fig = Figure()
ax = LScene(fig[1,1]; show_axis = false)
graphplot!(ax, g, layout=layout, node_size=0.0, edge_width=1.0)
display(fig)
##
observable(state) = (state[3] > 23) & (state[1] > 0)
o_t = [observable(state) for state in eachcol(timeseries)]
o_t2 = [observable(state) for state in eachcol(s_timeseries)]
o_t = [o_t..., o_t2...]
leaf_inds = P5
markov_states = zeros(3, length(P5))
for key in keys(P4)
    markov_states[:, P4[key]] .= CC[key]
end
o_e = [observable(state) for state in eachcol(markov_states)]
o_e2 = [observable(state) for state in centers]

ps = [length(f)/n for f in F]
time_average = mean(o_t)
ensemble_average = sum(o_e .* p)
ensemble_average2 = sum(o_e2 .* p2)
e1 = (time_average - ensemble_average) / time_average
e2 = (time_average - ensemble_average2) / time_average

println("The error for the uniform tree is $e1")
println("The error for the power tree is $e2")
##
E = eigen(Q)
E2 = eigen(Q2)
##
tmp = autocovariance(o_t[1:10:end]; timesteps = 1000)

tmp_e = autocovariance(o_e, E, collect(0:10:10000) * dt; progress=false)
tmp_e2 = autocovariance(o_e2, E2, collect(0:10:10000) * dt2; progress=false)
##
fig = Figure()
ax = Axis(fig[1,1])
lines!(ax, tmp, color = :black)
scatter!(ax, tmp_e, color = (:red, 0.5))
scatter!(ax, tmp_e2, color = (:blue, 0.5))
display(fig)