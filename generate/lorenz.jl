@info "evolving lorenz equations"

timesteps = 10^8
m_timeseries, Δt = lorenz_data(timesteps=timesteps, Δt=1e-3, ϵ = 0.0)
s_timeseries = lorenz_symmetry(m_timeseries)
joined_timeseries = hcat(m_timeseries, s_timeseries) # only for Partitioning Purpose
##
@info "saving data"
hfile = h5open(data_directory  * "/lorenz.hdf5", "w")
hfile["timeseries"] = m_timeseries
hfile["symmetrized timeseries"] = s_timeseries
hfile["dt"] = Δt
close(hfile)
