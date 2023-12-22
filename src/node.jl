
struct Node{P, C, D} 
    parent :: P
    children :: C
    data :: D
end

const EndNode = Node{<:Any, <:Nothing}
const StartNode = Node{<:Nothing, <:Any}

struct UnbalancedTreeEmbedding{N}
    nodes :: N
end

function UnbalancedTreeEmbedding(data; partitions = 2, max_levels = 20, min_probablility = 0.02)

    parents  = [data]
    
    level = 1
    probabilities = ones(partitions)

    while all(probabilities <= min_probablility)
        centers, children = AttractorConvergence.linear_split(parents[1], partitions)
        probabilities = 

    end

    return UnbalancedTreeEmbedding(nodes)
end

function UnbalancedTree()