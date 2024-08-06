dt_skip = 0.0001

@info "loading data"
hfile = h5open(data_directory  * "/lorenz.hdf5", "r")
dt = read(hfile["dt"])
skip = maximum([round(Int, dt_skip/dt), 1])
m_timeseries = read(hfile["timeseries"])[:, 1:skip:end]
s_timeseries = read(hfile["symmetrized timeseries"])[:, 1:skip:end]
joined_timeseries = hcat(m_timeseries, s_timeseries) # only for Partitioning Purpose
close(hfile)


hfile = h5open(data_directory  * "/time_mean_statistics.hdf5", "w")
hfile["skip"] = skip
hfile["dt_skip"] = dt_skip
for (jj, state_string) in ProgressBar(enumerate(["x", "y", "z"]))
    observables = [i -> i[jj]^j for j in 1:5]
    @info "computing observables from timeseries"
    observables_list = zeros(length(observables))
    for j in ProgressBar(eachindex(observables))
        N = length(eachcol(joined_timeseries))
        for state in ProgressBar(eachcol(joined_timeseries))
            observables_list[j] += observables[j](state) / N
        end
    end


    cumulants_list = similar(observables_list)
    cumulants_list[1] = observables_list[1]
    cumulants_list[2] = observables_list[2] - observables_list[1]^2
    cumulants_list[3] = observables_list[3] - 3 * observables_list[1] * observables_list[2] + 2 * observables_list[1]^3
    cumulants_list[4] = observables_list[4] - 4 * observables_list[1] * observables_list[3] - 3 * observables_list[2]^2 + 12 * observables_list[1]^2 * observables_list[2] - 6 * observables_list[1]^4
    cumulants_list[5] = observables_list[5] - 5 * observables_list[1] * observables_list[4] - 10 * observables_list[2] * observables_list[3] + 20 * observables_list[1]^2 * observables_list[3] + 30 * observables_list[1] * observables_list[2]^2 - 60 * observables_list[1]^3 * observables_list[2] + 24 * observables_list[1]^5

    hfile[state_string * " moments"] = observables_list[1:5]
    hfile[state_string * " cumulants"] = cumulants_list[1:5]
end
close(hfile)