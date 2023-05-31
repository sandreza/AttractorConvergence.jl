using ParallelKMeans
using ParallelKMeans: kmeans

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

splittableparents(c::Matrix) = length(c)

function ProbabilityTreeEmbedding(data; statesnumber = 100, reduction = 0.9, partitions = 2, split_function = linear_split)

    datasize = infer_datasize(data) 
    parents  = [data]
    prob     = datasize / statesnumber * reduction
    states   = []
    conn     = []
    bsize    = []

    splittable = [1]
    
    while length(states) <= statesnumber
        for parentindex in splittable
            centers, children = split_function(parents[parentidx], partitions)
            splittable = splittable(centers)
           
        end


    end

    treestates = GeneralizedTree{L}(states, conn, bsize)

    return ProbabilityTreeEmbedding(states)
end
