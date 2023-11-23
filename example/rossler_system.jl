using AttractorConvergence, ProgressBars, GLMakie, ParallelKMeans
using LinearAlgebra, MarkovChainHammer

function rossler(x⃗, t; a = 0.2, b = 0.2, c = 5.7)
    x = x⃗[1]
    y = x⃗[2]
    z = x⃗[3]
    ẋ = -y - z
    ẏ = x + a*y
    ż = b + z*(x-c)
    return [ẋ, ẏ, ż]
end

function thomas(x⃗, t; b = 0.208186)
    x = x⃗[1]
    y = x⃗[2]
    z = x⃗[3]
    ẋ = sin(y) - b * x
    ẏ = sin(z) - b * y
    ż = sin(x) - b * z
    return [ẋ, ẏ, ż]
end

function halvorsen(x⃗, t; a = 1.4)
    x = x⃗[1]
    y = x⃗[2]
    z = x⃗[3]
    ẋ = -a*x - 4*y - 4*z - y^2
    ẏ = -a*y - 4*z - 4*x - z^2
    ż = -a*z - 4*x - 4*y - x^2
    return [ẋ, ẏ, ż]
end

Δt = 0.1
timesteps = 10^5
ϵ = 0.0
initial_condition = [-2.930966942544555, -4.210223701652402, 0.022143486855594634]
# initial_condition =  [-2.999593177994478, -1.194425530641643, -2.957006094628252]
# initial_condition = [ -4.212546650394001, -6.779958866418458, 1.6254010889925299]
rhs(x, t) = rossler(x, t) # rossler(x, t)
x_f = zeros(3, timesteps)
x_f[:, 1] .= initial_condition
evolve! = RungeKutta4(3)
for i in ProgressBar(2:timesteps)
    xOld = x_f[:, i-1]
    evolve!(rhs, xOld, Δt)
    if ϵ > 0.0
        𝒩 = randn(3)
        @inbounds @. x_f[:, i] = evolve!.xⁿ⁺¹ + ϵ * sqrt(Δt) * 𝒩
    else
        @inbounds @. x_f[:, i] = evolve!.xⁿ⁺¹
    end
end

scatter(x_f)

##
function split(timeseries, indices, n_min; numstates = 2)
    if length(indices) > n_min
        r0 = kmeans(view(timeseries, :, indices), numstates; max_iters=10^4)
        inds = [[i for (j, i) in enumerate(indices) if r0.assignments[j] == k] for k in 1:numstates]
        centers = [r0.centers[:, k] for k in 1:numstates]
        return inds, centers
    end
    return [[]], [[]]
end

function unstructured_tree(timeseries, p_min; threshold = 2)
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
        inds, centers = split(timeseries, w, n_min)
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

struct UnstructuredTree{L, C, CH}
    leafmap::L 
    centers::C 
    children::CH
end

UnstructuredTree() = UnstructuredTree([], [], [])

function (embedding::UnstructuredTree)(state)
    current_index = 1
    while length(embedding.centers[current_index]) > 1
        local_child = argmin([norm(state - center) for center in embedding.centers[current_index]])
        current_index = embedding.children[current_index][local_child]
    end
    return embedding.leafmap[current_index]
end
##
# partition_state_space(timeseries; method = UnstructuredTree(probability_minimum = 0.1))
# partition_state_space(timeseries; method = StructuredTree(levels = 3, splitting = 2))
# partition_state_space(timeseries; method = KMeans(centers = 100))
# partition_state_space(timeseries; method = RandomPoints(100))
# partition_state_space(timeseries; method = Custom())
function partition_state_space(timeseries; pmin = 0.01)
    @info "determine partitioning function "
    F, G, H, PI, P3, P4, C, CC, P5 = unstructured_tree(timeseries, pmin)
    embedding = UnstructuredTree(P4, C, P3)
    me = Int64[]
    @info "computing markov embedding"
    for state in ProgressBar(eachcol(timeseries))
        push!(me, embedding(state))
    end
    return me
end
##
me = partition_state_space(x_f; pmin = 0.001/3)
##
P = perron_frobenius(me)
Q = generator(me, dt = Δt)
K = koopman_modes(P)
p = steady_state(P)
scaled_entropy(p)
km1 = abs.(K[end-1, :])
km2 = atan.(real.(K[end-1, :]), imag.(K[end-1, :]))
km3 = real.(K[end-3, :])
##
km1_colors = Float64[]
km2_colors = Float64[]
km3_colors=  Float64[]
for i in me
    push!(km1_colors, km1[i])
    push!(km2_colors, km2[i])
    push!(km3_colors, km3[i])
end
markov_indices = me
##
fig = Figure()
ax = LScene(fig[1,1]; show_axis=false)
lines!(ax, x_f, color=markov_indices, colormap=:glasbey_hv_n256, markersize=10)
ax = LScene(fig[1,2]; show_axis=false)
scatter!(ax, x_f, color=-km1_colors, colormap=:viridis, markersize=10)
ax = LScene(fig[2,1]; show_axis=false)
scatter!(ax, x_f, color=km2_colors, colormap=:balance, colorrange = (-π, π), markersize=10)
ax = LScene(fig[2,2]; show_axis=false)
scatter!(ax, x_f, color=km3_colors, colormap=:balance, colorrange = (-0.07, 0.07), markersize=10)
display(fig)

##
StateSpacePartitions.jl

# abstractembedding
struct StateSpacePartition{E, P}
    embedding::E 
    partition::P 
end

function StateSpacePartition(timeseries; pmin = 0.01)
    @info "determine partitioning function "
    F, G, H, PI, P3, P4, C, CC, P5 = unstructured_tree(timeseries, pmin)
    embedding = UnstructuredTree(P4, C, P3)
    me = Int64[]
    @info "computing markov embedding"
    for state in ProgressBar(eachcol(timeseries))
        push!(me, embedding(state))
    end
    return StateSpacePartition(embedding, me)
end

timeseries_partition = StateSpacePartition(x_f)

function visualize(timeseries, partition)
    if size(timeseries)[1] < 4
        fig = Figure()
        ax = LScene(fig[1,1]; show_axis=false)
        scatter!(ax, timeseries, color=partition, colormap=:glasbey_hv_n256, markersize=10)
        display(fig)
        return fig
    else 
        println("dimensions too high for visualization")
        return false 
    end
end

visualize(timeseries, partition::StateSpacePartition) = visualize(timeseries, partition.partition)

##
visualize(x_f, timeseries_partition)