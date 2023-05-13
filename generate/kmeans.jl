@info "starting k-means"
X = joined_timeseries[:, 1:1:end]
##
function split(X)
    numstates = 2
    r0 = kmeans(X, numstates; max_iters=10000)
    child_0 = (r0.assignments .== 1)
    child_1 = (!).(child_0)
    children = [view(X, :, child_0), view(X, :, child_1)]
    return r0.centers, children
end
level_global_indices(level) = 2^(level-1):2^level-1
##
levels = 10
parent_views = []
centers_list = []
push!(parent_views, X)
## Level 1
centers, children = split(X)
push!(centers_list, [centers[:, 1], centers[:, 2]])
push!(parent_views, children[1])
push!(parent_views, children[2])
## Levels 2 through levels
for level in ProgressBar(2:levels)
    for parent_global_index in level_global_indices(level)
        centers, children = split(parent_views[parent_global_index])
        push!(centers_list, [centers[:, 1], centers[:, 2]])
        push!(parent_views, children[1])
        push!(parent_views, children[2])
    end
end
@info "done with k-means"