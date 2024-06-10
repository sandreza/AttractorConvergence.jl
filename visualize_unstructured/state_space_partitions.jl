@info "loading data"
hfile = h5open(data_directory  * "/lorenz.hdf5", "r")
m_timeseries = read(hfile["timeseries"])
s_timeseries = read(hfile["symmetrized timeseries"])
joined_timeseries = hcat(m_timeseries, s_timeseries) # only for Partitioning Purpose
close(hfile)

hfile = h5open(data_directory  * "/embedding.hdf5", "r")
markov_chain = read(hfile["markov_chain"])
coarse_markov_chain = read(hfile["coarse_markov_chains"])
probability = read(hfile["probability"])
coarse_probabilities = read(hfile["coarse_probabilities"])
close(hfile)

##
inds = 1:10:size(joined_timeseries)[2]
fig = Figure(resolution=(2000, 2000))
for i in 1:9
    ii = (i - 1) ÷ 3 + 1
    jj = (i - 1) % 3 + 1
    ax = LScene(fig[ii, jj]; show_axis=false)
    markov_indices = coarse_markov_chain[inds, i]
    scatter!(ax, joined_timeseries[:, inds], color=markov_indices, colormap=:glasbey_hv_n256, markersize=5)
    rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
end
display(fig)