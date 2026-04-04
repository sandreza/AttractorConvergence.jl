using HDF5, MarkovChainHammer, StateSpacePartitions, ProgressBars, LinearSolve, IncompleteLU

data_directory = "/storage4/andre/attractor_convergence/real_data/"
hfile = h5open(data_directory * "ensemble_mean_statistics.hdf5", "r");
zmoments = zeros(5, 25)
for i in 1:25
    zmoments[:, i] = read(hfile["z moments $i"] )
end
close(hfile)

hfile = h5open(data_directory * "eigenvalues.hdf5", "r");
ps = Vector{Float64}[]
for i in 1:25
    push!(ps, read(hfile["generator steady state $i"]))
end
ps100 = Vector{Float64}[]
for i in 1:25
    push!(ps100, read(hfile["perron_frobenius 100 steady state $i"]))
end
close(hfile)

hfile = h5open(data_directory * "embedding.hdf5", "r");
mc22 = read(hfile["coarse_markov_chains 22"]);
mc23 = read(hfile["coarse_markov_chains 23"]);
close(hfile)

hfile = h5open(data_directory * "centers.hdf5", "r");
centers22 = read(hfile["centers 22"])
centers23 = read(hfile["centers 23"])
close(hfile)

hfile = h5open(data_directory * "lorenz.hdf5", "r");
trajectory = read(hfile["timeseries"]);
close(hfile)

##
index = 22 
num_partitions = length(ps[index])
new_cell_centers = zeros(3, num_partitions)
cell_count = zeros(Int, num_partitions)
cell_index = mc22
for i in ProgressBar(1:size(trajectory)[2])
    new_cell_centers[:, cell_index[i]] .+= trajectory[:, i]
    cell_count[cell_index[i]] += 1
end

p_exact = cell_count ./ sum(cell_count)
for i in 1:num_partitions
    new_cell_centers[:, i] .= new_cell_centers[:, i] / cell_count[i]
end
##
zmoments_index = zmoments[:, index]

moment_check = copy(zmoments_index) * 0
for j in ProgressBar(1:5)
    for i in 1:num_partitions
        moment_check[j] += new_cell_centers[3, i]^j * ps[index][i]
    end
end

relerror = (zmoments_index - moment_check) ./ zmoments_index 
println("Relative error: ", relerror)
