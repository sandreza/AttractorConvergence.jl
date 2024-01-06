function graph_from_edge_information(edge_information)
    N = maximum([maximum([edge_information[i][1], edge_information[i][2]]) for i in eachindex(edge_information)])
    adj = spzeros(Int64, N, N)
    adj_mod = spzeros(Float64, N, N)
    for i in ProgressBar(eachindex(edge_information))
        ii = edge_information[i][1]
        jj = edge_information[i][2]
        modularity_value = edge_information[i][3]
        adj[ii, jj] += 1
        adj_mod[ii, jj] = modularity_value
    end 
    N = maximum([maximum([edge_information[i][1], edge_information[i][2]]) for i in eachindex(edge_information)])
    node_labels = zeros(N)
    for i in eachindex(edge_information)
        node_labels[edge_information[i][2]] = edge_information[i][3]
    end
    node_labels[1] = 1.0
    return node_labels, adj, adj_mod, length(edge_information)
end

function isancestor(ancestor, child, G)
    for i in eachindex(G.badjlist)
        if ancestor == G.badjlist[child][1]
            return true
        else
            child = G.badjlist[child][1]
        end
        if isempty(G.badjlist[child])
            return false 
        end
    end
    return nothing
end

function leaf_global_to_local_indices(graph_edges)
    parents = [graph_edges[i][1] for i in eachindex(graph_edges)]
    children = [graph_edges[i][2] for i in eachindex(graph_edges)]
    childless = setdiff(children, parents)
    global_to_local = Dict()
    local_to_global = Dict()
    j = 0
    for i in eachindex(childless) 
        j += 1 
        global_to_local[childless[i]] = j
        local_to_global[j] = childless[i]
    end
    return global_to_local, local_to_global
end

# This needs improvement, still a bit too slow
function new_unstructured_coarsen_edges(graph_edges, probability_minimum, parent_to_children, G, global_to_local)
    parents = [graph_edges[i][1] for i in eachindex(graph_edges)]
    children = [graph_edges[i][2] for i in eachindex(graph_edges)]
    probabilities = [graph_edges[i][3] for i in eachindex(graph_edges)]
    parent_probabilities = Dict() 
    for parent in union(parents)
        parent_probabilities[parent] = 0
    end
    for i in eachindex(parents)
        parent_probabilities[parents[i]] += probabilities[i]
    end
    delete_children_list = Int64[]
    childless_list = Int64[]
    for parent in union(parents)
        if parent_probabilities[parent] < probability_minimum
            push!(childless_list, parent)
            push!(delete_children_list, parent_to_children[parent]...)
        end
    end
    new_parent_to_children = copy(parent_to_children)
    for child in delete_children_list
        delete!(new_parent_to_children, child)
    end
    for new_childless in childless_list
        new_parent_to_children[new_childless] = NaN
    end
    # The code below was generated
    new_graph_edges = Tuple{Int64, Int64, Float64}[]
    for i in eachindex(graph_edges)
        if !(graph_edges[i][1] in childless_list)
            push!(new_graph_edges, graph_edges[i])
        end
    end
    # The code above was generate
    # global_to_local, local_to_global = leaf_global_to_local_indices(graph_edges)
    local_to_global = Dict(value => key for (key, value) in global_to_local)
    new_global_to_local, new_local_to_global = leaf_global_to_local_indices(new_graph_edges)
    global_to_global = Dict()
    for key in delete_children_list
        for ancestor_key in keys(new_global_to_local)
            if isancestor(ancestor_key, key, G)
                global_to_global[key] = ancestor_key
                break
            end 
        end 
    end
    other_keys = setdiff(keys(global_to_local), delete_children_list)
    for key in other_keys
        global_to_global[key] = key
    end
    local_to_local = Dict()
    for key in keys(local_to_global)
        local_to_local[key] = new_global_to_local[global_to_global[local_to_global[key]]]
    end
    return local_to_local, new_local_to_global
end

function find_leaves_p2c(ind, parent_to_children)
    W = copy(parent_to_children[ind])
    local_leaves = Int64[]
    if length(W) == 1
        return [ind]
    else
        while length(W) > 0
            w = popfirst!(W)
            inds = copy(parent_to_children[w])
            if length(inds) > 1
                W = [inds..., W...]
            else
                push!(local_leaves, w)
            end
        end
    end
    return local_leaves
end

function new_unstructured_coarsening_p2c(p_min, parent_to_children, probabilities, global_to_local)
    coarse_global_to_local = Dict{Int64, Int64}()
    coarse_local_to_global = Dict{Int64, Int64}()
    W = copy(parent_to_children[1])
    leaf_index = 0
    # tic = Base.time()
    while length(W) > 0
        w = popfirst!(W)
        inds = parent_to_children[w]
        if length(inds) < 2
            leaf_index += 1
            coarse_global_to_local[w] = leaf_index
            coarse_local_to_global[leaf_index] = w
        else
            for ind in inds
                if probabilities[ind] > p_min
                    W = [ind, W...]
                else
                    leaf_index += 1
                    coarse_global_to_local[ind] = leaf_index
                    coarse_local_to_global[leaf_index] = ind
                end
            end
        end
    end
    # toc = Base.time()
    # println("took ", toc - tic, " seconds for finding root nodes")
    # @info "coarse graining map"
    # tic = Base.time()
    local_to_local = Dict{Int64, Int64}()
    for global_key in keys(coarse_global_to_local)
        local_leaves = find_leaves_p2c(global_key, parent_to_children)
        for leave in local_leaves
            local_to_local[global_to_local[leave]] = coarse_global_to_local[global_key]
        end
    end
    # toc = Base.time()
    # println("took ", toc - tic, " seconds for coarse graining map")
    return local_to_local, coarse_local_to_global
end