@info "loading data"
gap = 100
hfile = h5open(data_directory  * "/lorenz.hdf5", "r")
m_timeseries = read(hfile["timeseries"])[:, 1:gap:end]
s_timeseries = read(hfile["symmetrized timeseries"])[:, 1:gap:end]
joined_timeseries = hcat(m_timeseries, s_timeseries) # only for Partitioning Purpose
close(hfile)

#=

hfile = h5open(data_directory  * "/koopman_timeseries.hdf5", "r")
gap = gap ÷ 10
koopman_timeseries = Vector{Float64}[]
for (j, i) in ProgressBar(enumerate([10, 14, 18, 20]))
    push!(koopman_timeseries, read(hfile["generator koopman timeseries $i"])[1:gap:end])
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
=#


hfile = h5open(data_directory  * "/koopman_timeseries.hdf5", "r")
gap = gap ÷ 10
koopman_timeseries = Vector{Float64}[]
for (j, i) in ProgressBar(enumerate(([12, 16, 20] .+5)))
    push!(koopman_timeseries, read(hfile["generator koopman timeseries $i"])[1:gap:end])
end
close(hfile)

set_theme!(backgroundcolor=:white)
xmax_ind = argmax(joined_timeseries[1, :])
fig = Figure(resolution=(300, 100) .* 4 )
for i in 1:3
    ii = (i - 1) ÷ 3 + 1
    jj = (i - 1) % 3 + 1
    ax = LScene(fig[ii, jj]; show_axis=false)
    koopman_mode = koopman_timeseries[i]
    koopman_mode .*= sign.(koopman_mode[xmax_ind])
    upper_quantile = quantile(koopman_mode, 0.96)
    scatter!(ax, joined_timeseries, color=koopman_mode, colormap=:balance, markersize=5, colorrange = (-upper_quantile, upper_quantile))
    rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
end

# save("unstructured_figures" * "/Figure2.png", fig)