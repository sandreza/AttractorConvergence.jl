@info "evolving lorenz equations"

timesteps = 10^6
m_timeseries, Δt = lorenz_data(timesteps=timesteps, Δt=1e-2, ϵ = 0.0)
s_timeseries = lorenz_symmetry(m_timeseries)
##
@info "saving data"
hfile = h5open(data_directory  * "/lorenz_revision.hdf5", "w")
hfile["timeseries"] = m_timeseries
hfile["symmetrized timeseries"] = s_timeseries
hfile["dt"] = Δt
close(hfile)
##
timesteps = 5 * 10^5
m_timeseries, Δt = lorenz_data(timesteps=timesteps, Δt=2e-2, ϵ = 0.0)
s_timeseries = lorenz_symmetry(m_timeseries)
##
@info "saving data"
hfile = h5open(data_directory  * "/lorenz_revision_coarse.hdf5", "w")
hfile["timeseries"] = m_timeseries
hfile["symmetrized timeseries"] = s_timeseries
hfile["dt"] = Δt
close(hfile)