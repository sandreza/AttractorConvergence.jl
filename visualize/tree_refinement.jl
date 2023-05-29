levels = round(Int, log2(maximum(union(markov_chain))))
# set_theme!(backgroundcolor=:black)
fig = Figure(resolution=(2000, 2000))
for i in 1:9
    ii = (i - 1) ÷ 3 + 1
    jj = (i - 1) % 3 + 1
    ax = LScene(fig[ii, jj]; show_axis=false)
    markov_indices = div.(markov_chain[inds] .+ 2^levels .- 1, 2^(levels - i))
    scatter!(ax, m_timeseries[:, inds], color=markov_indices, colormap=:glasbey_hv_n256, markersize=10)
    rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
end
display(fig)