using HDF5, ProgressBars

# data_directory = "/storage4/andre/attractor_convergence" * "/real_data"
hfile = h5open(data_directory  * "/ensemble_mean_statistics.hdf5", "w")
centers_hfile = h5open(data_directory  * "/centers.hdf5", "r")
eigenvalues_hfile = h5open(data_directory  * "/eigenvalues.hdf5", "r")
imax = length(keys(centers_hfile)) - 1
for i in ProgressBar(1:imax)
    centers = read(centers_hfile["centers $i"])
    p = read(eigenvalues_hfile["generator steady state $i"] )
    hfile["number of partitions $i"] = length(p)
    for (jj, state_string) in ProgressBar(enumerate(["x", "y", "z"]))
        observables = [i -> i[jj]^j for j in 1:5]
        partition_number = size(centers)[2]
        ensemble_mean = zeros(length(observables))
        for (oo,observable) in enumerate(observables)
            for ii in 1:partition_number
                ensemble_mean[oo] += observable(centers[:, ii]) .* p[ii]
            end
        end
        hfile[state_string * " moments $i"] = ensemble_mean
    end
end
close(centers_hfile)
close(eigenvalues_hfile)
close(hfile)

##
hfile = h5open(data_directory  * "/ensemble_mean_statistics.hdf5", "r+")
for i in ProgressBar(1:imax)
    for state_string in ["x", "y", "z"]
        moments = read(hfile[state_string * " moments $i"])
        cumulants_list = similar(moments)
        cumulants_list[1] = moments[1]
        cumulants_list[2] = moments[2] - moments[1]^2
        cumulants_list[3] = moments[3] - 3 * moments[1] * moments[2] + 2 * moments[1]^3
        cumulants_list[4] = moments[4] - 4 * moments[1] * moments[3] - 3 * moments[2]^2 + 12 * moments[1]^2 * moments[2] - 6 * moments[1]^4
        cumulants_list[5] = moments[5] - 5 * moments[1] * moments[4] - 10 * moments[2] * moments[3] + 20 * moments[1]^2 * moments[3] + 30 * moments[1] * moments[2]^2 - 60 * moments[1]^3 * moments[2] + 24 * moments[1]^5
        hfile[state_string * " cumulants $i"] = cumulants_list
    end
end
close(hfile)