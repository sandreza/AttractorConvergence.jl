F[2744]




using HDF5
using ParallelKMeans, NetworkLayout, Graphs, Printf, GraphMakie
using HDF5, GraphMakie, NetworkLayout, MarkovChainHammer, ProgressBars, GLMakie, Graphs
using Printf, Random, SparseArrays

Random.seed!(12345)

function split3(timeseries, indices, pmin; numstates = 2)
    tmp = ones(Int, n)
    tmp[indices] .+= 1
    P = perron_frobenius(tmp)
    if P[2,2] > pmin
        r0 = kmeans(view(timeseries, :, indices), numstates; max_iters=10^4)
        inds = [[i for (j, i) in enumerate(indices) if r0.assignments[j] == k] for k in 1:numstates]
        centers = [r0.centers[:, k] for k in 1:numstates]
        return inds, centers
    end
    return [[]], [[]]
end

function unstructured_tree3(timeseries, p_min)
    n = size(timeseries)[2]
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
        inds, centers = split3(timeseries, w, p_min)
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

# To do list 
# given parent index, output children 
# given leaf node global index, output local index 
# write embedding function 
# compare with power_tree.jl
##
@info "constructing data"
kmeans_data = timeseries # hcat(timeseries[:, 1:100:end], s_timeseries[:, 1:100:end])
@info "constructing tree"
F, G, H, PI, P3, P4, C, CC, P5 = unstructured_tree3(kmeans_data, 0.92);
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
println("scaled entropy $se and ratio $pr for a size $(length(c))")
##
embedding2 = UnstructuredTree(P4, C, P3)
me = Int64[]
@info "computing markov embedding"
for state in ProgressBar(eachcol(timeseries))
    push!(me, embedding2(state))
end
##
#=
@info "computing symmetric embedding"
me_s = Int64[]
for state in ProgressBar(eachcol(s_timeseries))
    push!(me_s, embedding(state))
end
=#
P = perron_frobenius(me)

scatter(sort([P[i,i] for i in 1:size(P)[1]]))
scatter(sort(steady_state(P)))

Q = generator(me; dt = dt)
Λ, V =  eigen(P)
W4 = koopman_modes(P)
##
koopman_mode = real.(W4[end-3, :])
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