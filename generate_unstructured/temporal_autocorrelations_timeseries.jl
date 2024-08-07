using HDF5, Statistics, MarkovChainHammer

dt_skip = 0.01
@info "loading data"
hfile = h5open(data_directory  * "/lorenz.hdf5", "r")
dt = read(hfile["dt"])
skip = maximum([round(Int, dt_skip/dt), 1])
m_timeseries = read(hfile["timeseries"])[:, 1:skip:end]
close(hfile)

ztimeseries = m_timeseries[3, :]

##
Tend = 40
timesteps = round(Int, Tend/dt_skip)
ac = autocovariance(ztimeseries; timesteps = timesteps, progress =true)

hfile = h5open(data_directory  * "/temporal_autocovariance.hdf5", "w")
hfile["time mean autocovariance"] = ac
close(hfile)
hfile = h5open(data_directory  * "/temporal_autocovariance.hdf5", "r+")
hfile["dt_skip"] = dt_skip
close(hfile)