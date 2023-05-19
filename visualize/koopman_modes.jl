

levels = round(Int, log2(maximum(union(markov_chain))))
inds = 1:5:10^7
# left_eigenvectors_list[end][1, :]
fig = Figure(resolution=(2000, 2000))
set_theme!(backgroundcolor=:black)
for i in 1:4
    ii = (i - 1) ÷ 2 + 1
    jj = (i - 1) % 2 + 1
    level = i + 7
    ax = LScene(fig[ii, jj]; show_axis=false)
    koopman_mode = real.(left_eigenvectors_list[level][end-3, :])
    koopman_mode = koopman_mode ./ sign(koopman_mode[end])
    markov_indices = div.(markov_chain[inds] .+ 2^levels .- 1, 2^(levels - level)) .- minimum(div.(markov_chain[inds] .+ 2^levels .- 1, 2^(levels - level))) .+1
    colors = [koopman_mode[markov_indices[j]] for j in 1:length(markov_indices)]
    scatter!(ax, m_timeseries[:, inds]; color=colors, colormap=:balance, markersize=3)
    rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
end
display(fig)