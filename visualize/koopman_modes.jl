

levels = round(Int, log2(maximum(union(markov_chain))))
# left_eigenvectors_list[end][1, :]
fig = Figure(resolution=(2000, 2000))
set_theme!(backgroundcolor=:black)
markersize = 5
for i in 1:4
    ii = (i - 1) ÷ 2 + 1
    jj = (i - 1) % 2 + 1
    level = i + 7
    ax = LScene(fig[ii, jj]; show_axis=false)
    koopman_mode = real.(left_eigenvectors_list[level][end-3, :])
    koopman_mode = koopman_mode ./ sign(koopman_mode[end])
    markov_indices = div.(markov_chain[inds] .+ 2^levels .- 1, 2^(levels - level)) .- minimum(div.(markov_chain[inds] .+ 2^levels .- 1, 2^(levels - level))) .+1
    colors = [koopman_mode[markov_indices[j]] for j in 1:length(markov_indices)]
    scatter!(ax, m_timeseries[:, inds]; color=colors, colormap=:balance, markersize=markersize)
    rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
end
display(fig)

##
level = 11
koopman_mode = real.(left_eigenvectors_list[level][end-3, :])
koopman_mode = koopman_mode ./ sign(koopman_mode[end])
markov_indices = div.(markov_chain[inds] .+ 2^levels .- 1, 2^(levels - level)) .- minimum(div.(markov_chain[inds] .+ 2^levels .- 1, 2^(levels - level))) .+ 1
colors = [koopman_mode[markov_indices[j]] for j in 1:length(markov_indices)]
blue_quant = quantile(colors, 0.35)
red_quant = quantile(colors, 0.65)
white_region = findall(x -> blue_quant < x < red_quant, colors)
blue_region = findall(x -> x < blue_quant, colors)
red_region = findall(x -> x > red_quant, colors)
##
fig2 = Figure() 
ax = LScene(fig2[1, 1]; show_axis=false)
# scatter!(ax, m_timeseries[:, inds]; color=colors, colormap=:balance, markersize=markersize)
scatter!(ax, m_timeseries[:, inds[blue_region]]; color= :blue, markersize=markersize)
scatter!(ax, m_timeseries[:, inds[red_region]]; color=:red, markersize=markersize)
scatter!(ax, m_timeseries[:, inds[white_region]]; color=:white, markersize=markersize)
rotate_cam!(ax.scene, (0.0, -10.5, 0.0))

ax = LScene(fig2[1, 2]; show_axis=false)
scatter!(ax, m_timeseries[:, inds[blue_region]]; color= :blue, markersize=markersize)
rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
ax = LScene(fig2[2, 1]; show_axis=false)
scatter!(ax, m_timeseries[:, inds[red_region]]; color=:red, markersize=markersize)
rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
ax = LScene(fig2[2, 2]; show_axis=false)
scatter!(ax, m_timeseries[:, inds[white_region]]; color=:white, markersize=markersize)
rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
display(fig2)

##
fig3 = Figure() 
for i in 1:4
    ii = (i - 1) ÷ 2 + 1
    jj = (i - 1) % 2 + 1
    level = i + 8
    koopman_mode = real.(left_eigenvectors_list[level][end-3, :])
    koopman_mode = koopman_mode ./ sign(koopman_mode[end])
    markov_indices = div.(markov_chain[inds] .+ 2^levels .- 1, 2^(levels - level)) .- minimum(div.(markov_chain[inds] .+ 2^levels .- 1, 2^(levels - level))) .+ 1
    colors = [koopman_mode[markov_indices[j]] for j in 1:length(markov_indices)]
    blue_quant = quantile(colors, 0.35)
    red_quant = quantile(colors, 0.65)
    white_region = findall(x -> blue_quant < x < red_quant, colors)
    blue_region = findall(x -> x < blue_quant, colors)
    red_region = findall(x -> x > red_quant, colors)
    ax = LScene(fig3[ii, jj]; show_axis=false)
    # scatter!(ax, m_timeseries[:, inds]; color=colors, colormap=:balance, markersize=markersize)
    scatter!(ax, m_timeseries[:, inds[blue_region]]; color= :blue, markersize=markersize)
    scatter!(ax, m_timeseries[:, inds[red_region]]; color=:red, markersize=markersize)
    scatter!(ax, m_timeseries[:, inds[white_region]]; color=:white, markersize=markersize)
    rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
end
display(fig3)