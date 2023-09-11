using LinearAlgebra, SparseArrays

function modularity_matrix(A)
    N = size(A)[1]
    b = zeros(N, N)
    for i = 1:N, j = 1:N
        b[i, j] = (A[i, j] - (1 - A[i, i]) * (sum(A[j, :])) / N) / N
    end
    B = Symmetric(b + b')
    return B
end

function principal_vector(B::Symmetric)
    s = ones(Int, size(B)[1])
    Λ, V = eigen(B)
    v₁ = V[:, sortperm(real.(Λ))[end]]
    s[v₁.<=0] .= -1
    return s
end

function modularity(B, s)
    return s' * (B * s)
end

function modularity_eig(B)
    Λ, V = eigen(B)
    return maximum(Λ)
end

modularity(B::Symmetric) = modularity(B, principal_vector(B))

function split_community(B, indices, q_min)
    Bg = B[indices, :][:, indices]
    Bg = Bg - Diagonal(sum(Bg, dims=1)[:])
    Bg = Symmetric(Bg + Bg')
    s = principal_vector(Bg)
    q = modularity(Bg)
    qq = q_min

    if (q > q_min)
        ind1 = [i for (j, i) in enumerate(indices) if s[j] == 1]
        ind2 = [i for (j, i) in enumerate(indices) if s[j] == -1]
        qq = q
        return ind1, ind2, qq
    end
    return [], [], qq
end

function leicht_newman_with_tree(A, q_min::Float64)
    B = modularity_matrix(A)
    n = size(A)[1]
    W, F, G, P1, P2 = [collect(1:n)], [], [], [1], []
    qOld = 0.0
    H = []
    global_index = 1
    while (length(W) > 0)
        w = popfirst!(W)
        p1 = popfirst!(P1)
        ind1, ind2, q = split_community(B, w, q_min)
        if (length(ind1) > 0) & (length(ind2) > 0)
            W = [ind1, ind2, W...]
            P1 = [global_index + 1, global_index + 2, P1...]
            P2 = push!(P2, (p1, global_index + 1, q))
            P2 = push!(P2, (p1, global_index + 2, q))
            global_index += 2
            push!(H, [ind1, ind2, q])
            if q > 0
                qOld = q
            end
        else
            push!(F, w)
            push!(G, qOld)
        end
    end
    return F, G, H, P2
end

function graph_from_PI(PI)
    N = maximum([maximum([PI[i][1], PI[i][2]]) for i in eachindex(PI)])
    adj = spzeros(Int64, N, N)
    adj_mod = spzeros(Float64, N, N)
    for i in ProgressBar(eachindex(PI))
        ii = PI[i][1]
        jj = PI[i][2]
        modularity_value = PI[i][3]
        adj[ii, jj] += 1
        adj_mod[ii, jj] = modularity_value
    end 
    N = maximum([maximum([PI[i][1], PI[i][2]]) for i in eachindex(PI)])
    node_labels = zeros(N)
    for i in eachindex(PI)
        node_labels[PI[i][2]] = PI[i][3]
    end
    
    return node_labels, adj, adj_mod, length(PI)
end