@info "loading data"
skip = 1000
hfile = h5open(data_directory  * "/lorenz.hdf5", "r")
m_timeseries = read(hfile["timeseries"])[:, 1:skip:end]
s_timeseries = read(hfile["symmetrized timeseries"])[:, 1:skip:end]
joined_timeseries = hcat(m_timeseries, s_timeseries) # only for Partitioning Purpose
close(hfile)


hfile = h5open(data_directory  * "/koopman_timeseries.hdf5", "r")
skip = skip ÷ 10
koopman_timeseries = Vector{Float64}[]
for (j, i) in ProgressBar(enumerate([10, 14, 18, 20]))
    push!(koopman_timeseries, read(hfile["generator koopman timeseries $i"])[1:skip:end])
end
close(hfile)

##
set_theme!(backgroundcolor=:white)
xmax_ind = argmax(joined_timeseries[1, :])
fig = Figure(resolution=(1000, 1000))
for i in 1:4
    ii = (i - 1) ÷ 2 + 1
    jj = (i - 1) % 2 + 1
    ax = LScene(fig[ii, jj]; show_axis=false)
    koopman_mode = koopman_timeseries[i]
    koopman_mode .*= sign.(koopman_mode[xmax_ind])
    upper_quantile = quantile(koopman_mode, 0.9)
    scatter!(ax, joined_timeseries, color=koopman_mode, colormap=:balance, markersize=5, colorrange = (-upper_quantile, upper_quantile))
    rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
end