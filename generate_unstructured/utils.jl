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
