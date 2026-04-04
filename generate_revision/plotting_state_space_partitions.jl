using GLMakie, HDF5
data_directory = "./data"
@info "loading data"
hfile = h5open(data_directory  * "/lorenz_revision.hdf5", "r")
m_timeseries = read(hfile["timeseries"])
close(hfile)

hfile = h5open(data_directory  * "/structured_embedding_revision.hdf5", "r")
unstructured_chain = read(hfile["markov_chain_unstructured"])
kmeans_chain = read(hfile["markov_chain_kmeans"])
ulam_chain = read(hfile["markov_chain_ulam"])
close(hfile)
inds = 1:5000:size(m_timeseries)[2]
markov_chains = [unstructured_chain[inds], kmeans_chain[inds], ulam_chain[inds]]
##

jts = m_timeseries[:, inds]
fig = Figure(resolution=(2000, 2000))
ga  = GridLayout(fig[1, 1])
for i in 1:3
    ax = LScene(ga[1, i]; show_axis=false)
    markov_indices = markov_chains[i]
    scatter!(ax, jts, color=markov_indices, colormap=:glasbey_hv_n256, markersize=5)
    rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
end

colgap!(ga, 0.0)
rowgap!(ga, 0.0)
save("RevisionFigures/partitions.png", fig)