@info "loading data"
hfile = h5open(data_directory  * "/lorenz.hdf5", "r")
m_timeseries = read(hfile["timeseries"])
s_timeseries = read(hfile["symmetrized timeseries"])
joined_timeseries = hcat(m_timeseries, s_timeseries) # only for Partitioning Purpose
close(hfile)

hfile = h5open(data_directory  * "/embedding.hdf5", "r")
markov_chain = read(hfile["markov_chain"])
inds = 1:5000:size(joined_timeseries)[2]
coarse_markov_chain = zeros(Int, length(inds), 12)
for i in ProgressBar(1:12)
    coarse_markov_chain[:, i] .= read(hfile["coarse_markov_chains $i"])[inds]
end
close(hfile)

##
jts = joined_timeseries[:, inds]
fig = Figure(resolution=(2000, 2000))
for i in ProgressBar(1:9)
    ii = (i - 1) ÷ 3 + 1
    jj = (i - 1) % 3 + 1
    ax = LScene(fig[ii, jj]; show_axis=false)
    markov_indices = coarse_markov_chain[:, i+2]
    scatter!(ax, jts, color=markov_indices, colormap=:glasbey_hv_n256, markersize=5)
    rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
end
