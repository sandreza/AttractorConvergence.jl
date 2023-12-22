import Base

abstract type AbstractTree{L, T} end

@inline Base.size(tree::AbstractTree{L}) where L = size(tree, L)
@inline Base.size(tree::AbstractTree, L) = L == 0 ? 0 : sum(branch_size(tree, l) for l in 1:L) 

@inline Base.getindex(tree::AbstractTree, l, n)     = tree.data[globalindex(tree, l, n)]
@inline Base.setindex!(tree::AbstractTree, v, l, n) = setindex!(tree.data, v, globalindex(tree, l, n))

@inline globalindex(tree::AbstractTree, l, n)            = size(tree, l-1) + n
@inline globalindex(tree::AbstractTree, l, ::Colon)      = UnitRange(size(tree, l-1)+1, size(tree, l))
@inline globalindex(tree::AbstractTree, l, r::UnitRange) = size(tree, l-1) .+ r

@inline levels(::AbstractTree{L}) where L = L

struct PowerTree{N, L, T} <: AbstractTree{L, T} 
    data :: AbstractArray{T, 1}
    PowerTree{N, L}(data::AbstractArray{T, 1}) where {N, L, T} = new{N, L, T}(data)
end

@inline Base.summary(tree::PowerTree) = summary(tree.data)

@inline Base.show(io::IO, tree::PowerTree{N, L}) where {N, L} =
        print(io, "PowerTree with $N partitions per $L levels and data $(tree.data)")

PowerTree(datatype = Float64; partitions = 2, levels = 5) = 
        PowerTree{partitions, levels}(Vector{datatype}(undef, sum(Tuple(partitions^l for l in 1:levels))))

@inline branch_size(::PowerTree{N}, l) where N = N^l

@inline childindices(::PowerTree{N, L}, parentnode) where {N, L} = 1 + (parentnode - 1) * N : parentnode * N
@inline parentindex( ::PowerTree{N, L}, childnode)  where {N, L} = (childnode - 1) ÷ N + 1

flatten(tree::PowerTree) = flatten(tree.data)

struct GeneralizedTree{L, T, C, S} <: AbstractTree{L, T}
    data  :: AbstractArray{T, 1}
    conn  :: C
    bsize :: S

    GeneralizedTree{L}(data::AbstractArray{T, 1}, conn::C, bsize::S) where {L, T, C, S} = new{L, T, C, S}(data, conn, bsize)
end

@inline Base.summary(tree::GeneralizedTree) = summary(tree.data)

@inline Base.show(io::IO, tree::GeneralizedTree{L}) where {L} =
        print(io, "GenerlizedTree with $L levels and data $(tree.data)")

@inline branch_size(tree::GeneralizedTree, l) = tree.bsize[l]

function flatten(data::Vector{Vector{FT}}) where FT 
    newdata = zeros(FT, length(data), length(data[1]))
    for (i, d) in enumerate(data)
        newdata[i, :] .= d
    end
    return newdata
end

function buildup(data::AbstractArray{FT, 2}) where FT
    newdata = Array[]
    for i in 1:size(data, 2)
        push!(newdata, data[:, i])
    end
    return newdata
end

