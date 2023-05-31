using ParallelKMeans
using ParallelKMeans: kmeans
using Statistics: mean

abstract type AbstractEmbedding end

struct StateEmbedding{S} <: AbstractEmbedding
    markov_states::S
end

(embedding::StateEmbedding)(current_state) = 
    argmin([norm(current_state - markov_state) for markov_state in embedding.markov_states])

abstract type AbstractTreeEmbedding <: AbstractEmbedding end

struct PowerTreeEmbedding{N, L} <: AbstractTreeEmbedding 
    markov_states :: PowerTree{N, L}
end

childindices(embedding::PowerTreeEmbedding, parentindex) = childindices(embedding.markov_states, parentindex)
levels(embedding::PowerTreeEmbedding) = levels(embedding.markov_states)

function (embedding::PowerTreeEmbedding)(current_state)
    index = 1
    for l in 1:levels(embedding)
        search = childindices(embedding, index)
        index  = argmin([norm(current_state - markov_state) for markov_state in embedding.markov_states[l, search]])
        index  = search[index]
    end
    return index
end

euclidean_distance(x, y) = sqrt(sum((x .- y).^2))
maxk(a, k) = partialsortperm(a, 1:k, rev=true)

function even_split(data, partitions; max_iters = 10000, tolerance = 1.5)

    centers, children = linear_split(data, partitions; max_iters)
    children = buildup.(children)

    max_number = (size(data, 2) ÷ partitions) * tolerance
    min_number = (size(data, 2) ÷ partitions) / tolerance

    add_or_remove = zeros(Int, partitions)
    numelem       = [length(children[i]) for i in 1:partitions]
    for i in 1:partitions
        add_or_remove[i] = numelem[i] > max_number ? 2 : numelem[i] < min_number ? 1 : 0
    end

    if sum(add_or_remove) == 0 
        children = children .|> Vector{Vector{Float64}}
        children = flatten.(children)
    
        for i in 1:partitions
            children[i] = Array(children[i]')
        end
    
        return centers, children
    end

    maxidx = argmax(add_or_remove)
    
    distance = zeros(length(children[maxidx]))
    for i in eachindex(distance)
        distance[i] = euclidean_distance(children[maxidx][i], centers[:, maxidx])
    end

    toremove = numelem[maxidx] - minimum(numelem)
    rmidx    = maxk(distance, toremove)
    rmelem   = children[maxidx][rmidx]
    deleteat!(children[maxidx], sort(rmidx))
    
    for elem in rmelem
        ndist = [euclidean_distance(elem, centers[:, i]) for i in 1:partitions]
        perm  = sortperm(ndist)
        push!(children[perm[2]], elem)
    end

    centers  = Array(flatten(mean.(children))')
    children = children .|> Vector{Vector{Float64}}
    children = flatten.(children)

    for i in 1:partitions
        children[i] = Array(children[i]')
    end

    return centers, children
end

function linear_split(data, partitions; max_iters = 10000)
    kpartition = kmeans(data, partitions; max_iters)
    children   = []
    for center = 1:partitions
        child_indices = (kpartition.assignments .== center)
        push!(children, view(data, :, child_indices))
    end

    return kpartition.centers, children
end

infer_datatype(::Array{T, N}) where {T, N} = Array{T, N-1}
infer_datatype(::Array{T, 1}) where T      = T

function PowerTreeEmbedding(data; partitions = 2, levels = 5, split_function = linear_split)

    parents  = [data]
    datatype = infer_datatype(data) 
    states   = PowerTree(datatype; partitions, levels)
    
    for level in ProgressBar(1:levels)    
        cumulate_parents = []
        for parentidx in 1:branch_size(states, level - 1)
            centers, children = split_function(parents[parentidx], partitions)
            childidx = childindices(states, parentidx)
            for (i, child) in enumerate(childidx)
                states[level, child] = centers[:, i]
            end
            push!(cumulate_parents, children...)
        end
        parents = cumulate_parents
    end

    return PowerTreeEmbedding(states)
end

struct ProbabilityTreeEmbedding{L} <: AbstractTreeEmbedding 
    markov_states :: GeneralizedTree{L}
end

infer_datasize(a::Array{T, N}) where {T, N} = size(a, N)
infer_datasize(a::Array{T, 1}) where T      = length(a)

