
hfile = h5open(pwd() * "/data/lorenz.hdf5", "r")
timeseries = read(hfile["timeseries"])
s_timeseries = read(hfile["symmetrized timeseries"])
joined_timeseries = hcat(timeseries, s_timeseries) # only for Partitioning Purpose
close(hfile)
@info "starting k-means"
maxlength = minimum([10^7, size(joined_timeseries)[2] ])
inds = round.(Int,range(1, size(joined_timeseries)[2] + 1, length =maxlength+1))[1:end-1]
X = joined_timeseries[:, inds]
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
levels = 12
parent_views = []
centers_list = Vector{Vector{Float64}}[]
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
##
# save the centers 
centers_matrix = zeros(length(centers_list[1][1]), length(centers_list[1]), length(centers_list))
for i in eachindex(centers_list)
    centers_matrix[:, :, i] = hcat(centers_list[i]...)
end
##
@info "saving centers list data"
hfile = h5open(pwd() * "/data/kmeans.hdf5", "w")
hfile["centers"] = centers_matrix
hfile["levels"] = levels
close(hfile)
