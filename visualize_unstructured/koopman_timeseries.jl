using HDF5, GLMakie, ProgressBars, Statistics
@info "grabbing timeseries"
data_directory = "/storage4/andre/attractor_convergence" * "/real_data"
tic = Base.time()
hfile = h5open(data_directory  * "/lorenz.hdf5", "r")
m_timeseries = read(hfile["timeseries"])
s_timeseries = read(hfile["symmetrized timeseries"])
joined_timeseries = hcat(m_timeseries, s_timeseries) # only for Partitioning Purpose
close(hfile)

hfile = h5open(data_directory  * "/embedding.hdf5", "r")
coarse_probabilities = read(hfile["coarse_probabilities"])
close(hfile)
toc = Base.time() 
println("The amount of time that it took is $(toc - tic) seconds")

@info "grabbing data"
kthfile = h5open(data_directory  * "/koopman_timeseries.hdf5", "r")
q_modes = []
pf1_modes = []
pf10_modes = []
pf100_modes = []
skips = []
for (index, probability) in ProgressBar(enumerate(coarse_probabilities))
    koopman_timeseries  = read(kthfile["generator koopman timeseries $index"])
    skipdt = read(kthfile["generator timeseries dt $index"])
    skip = read(kthfile["generator timeseries skip $index"])
    push!(skips, skip)
    push!(q_modes, koopman_timeseries)
    for k in [1, 10, 100]
        pf_timeseries = read(kthfile["perron_frobenius $k koopman timeseries $index"])
        skipdt = read(kthfile["perron_frobenius $k timeseries dt $index"])
        skip = read(kthfile["perron_frobenius $k timeseries skip $index"])
        if k == 1
            push!(pf1_modes, pf_timeseries)
        elseif k == 10
            push!(pf10_modes, pf_timeseries)
        elseif k == 100
            push!(pf100_modes, pf_timeseries)
        else
            nothing
        end
    end
end
close(kthfile)

##
figure_directory = pwd() * "/unstructured_figures"
@info "plotting modes"
set_theme!(backgroundcolor=:white)
fig = Figure(resolution=(1000, 1000))
shift = 16
skip_more = 100
for i in 1:9
    ii = (i - 1) ÷ 3 + 1
    jj = (i - 1) % 3 + 1
    ax = LScene(fig[ii, jj]; show_axis=false)
    skip = skips[i+shift]
    inds = 1:skip*skip_more:size(joined_timeseries)[2]
    xmax_ind = argmax(joined_timeseries[1, inds])
    koopman_mode = q_modes[i + shift][1:skip_more:end]
    koopman_mode .*= sign.(koopman_mode[xmax_ind])
    upper_quantile = quantile(koopman_mode, 0.9)
    scatter!(ax, joined_timeseries[:, inds], color=koopman_mode, colormap=:balance, markersize=5, colorrange = (-upper_quantile, upper_quantile))
    rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
end
save(figure_directory * "/q_koopman.png", fig)

fig = Figure(resolution=(1000, 1000))
shift = 16
skip_more = 100
for i in 1:9
    ii = (i - 1) ÷ 3 + 1
    jj = (i - 1) % 3 + 1
    ax = LScene(fig[ii, jj]; show_axis=false)
    skip = skips[i+shift]
    inds = 1:skip*skip_more:size(joined_timeseries)[2]
    xmax_ind = argmax(joined_timeseries[1, inds])
    koopman_mode = pf1_modes[i + shift][1:skip_more:end]
    koopman_mode .*= sign.(koopman_mode[xmax_ind])
    upper_quantile = quantile(koopman_mode, 0.9)
    scatter!(ax, joined_timeseries[:, inds], color=koopman_mode, colormap=:balance, markersize=5, colorrange = (-upper_quantile, upper_quantile))
    rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
end
save(figure_directory * "/p1_koopman.png", fig)

fig = Figure(resolution=(1000, 1000))
shift = 16
skip_more = 100
for i in 1:9
    ii = (i - 1) ÷ 3 + 1
    jj = (i - 1) % 3 + 1
    ax = LScene(fig[ii, jj]; show_axis=false)
    skip = skips[i+shift]
    inds = 1:skip*skip_more:size(joined_timeseries)[2]
    xmax_ind = argmax(joined_timeseries[1, inds])
    koopman_mode = pf10_modes[i + shift][1:skip_more:end]
    koopman_mode .*= sign.(koopman_mode[xmax_ind])
    upper_quantile = quantile(koopman_mode, 0.9)
    scatter!(ax, joined_timeseries[:, inds], color=koopman_mode, colormap=:balance, markersize=5, colorrange = (-upper_quantile, upper_quantile))
    rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
end
save(figure_directory * "/p10_koopman.png", fig)

fig = Figure(resolution=(1000, 1000))
shift = 16
skip_more = 100
for i in 1:9
    ii = (i - 1) ÷ 3 + 1
    jj = (i - 1) % 3 + 1
    ax = LScene(fig[ii, jj]; show_axis=false)
    skip = skips[i+shift]
    inds = 1:skip*skip_more:size(joined_timeseries)[2]
    xmax_ind = argmax(joined_timeseries[1, inds])
    koopman_mode = pf100_modes[i + shift][1:skip_more:end]
    koopman_mode .*= sign.(koopman_mode[xmax_ind])
    upper_quantile = quantile(koopman_mode, 0.9)
    scatter!(ax, joined_timeseries[:, inds], color=koopman_mode, colormap=:balance, markersize=5, colorrange = (-upper_quantile, upper_quantile))
    rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
end
save(figure_directory * "/p100_koopman.png", fig)

##
fig = Figure(resolution=(1000, 1000))
shift = 24
skip_more = 10
for i in 1:1
    ii = (i - 1) ÷ 1 + 1
    jj = (i - 1) % 1 + 1
    ax = LScene(fig[ii, jj]; show_axis=false)
    skip = skips[i+shift]
    inds = 1:skip*skip_more:size(joined_timeseries)[2]
    xmax_ind = argmax(joined_timeseries[1, inds])
    koopman_mode = q_modes[i + shift][1:skip_more:end]
    koopman_mode .*= sign.(koopman_mode[xmax_ind])
    upper_quantile = quantile(koopman_mode, 0.97)
    scatter!(ax, joined_timeseries[:, inds], color=koopman_mode, colormap=:balance, markersize=5, colorrange = (-upper_quantile, upper_quantile))
    rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
end
save(figure_directory * "/q_koopman_zoom.png", fig)

fig = Figure(resolution=(1000, 1000))
shift = 24
skip_more = 10
for i in 1:1
    ii = (i - 1) ÷ 1 + 1
    jj = (i - 1) % 1 + 1
    ax = LScene(fig[ii, jj]; show_axis=false)
    skip = skips[i+shift]
    inds = 1:skip*skip_more:size(joined_timeseries)[2]
    xmax_ind = argmax(joined_timeseries[1, inds])
    koopman_mode = pf1_modes[i + shift][1:skip_more:end]
    koopman_mode .*= sign.(koopman_mode[xmax_ind])
    upper_quantile = quantile(koopman_mode, 0.97)
    scatter!(ax, joined_timeseries[:, inds], color=koopman_mode, colormap=:balance, markersize=5, colorrange = (-upper_quantile, upper_quantile))
    rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
end
save(figure_directory * "/p1_koopman_zoom.png", fig)

fig = Figure(resolution=(1000, 1000))
shift = 24
skip_more = 10
for i in 1:1
    ii = (i - 1) ÷ 1 + 1
    jj = (i - 1) % 1 + 1
    ax = LScene(fig[ii, jj]; show_axis=false)
    skip = skips[i+shift]
    inds = 1:skip*skip_more:size(joined_timeseries)[2]
    xmax_ind = argmax(joined_timeseries[1, inds])
    koopman_mode = pf10_modes[i + shift][1:skip_more:end]
    koopman_mode .*= sign.(koopman_mode[xmax_ind])
    upper_quantile = quantile(koopman_mode, 0.97)
    scatter!(ax, joined_timeseries[:, inds], color=koopman_mode, colormap=:balance, markersize=5, colorrange = (-upper_quantile, upper_quantile))
    rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
end
save(figure_directory * "/p10_koopman_zoom.png", fig)

fig = Figure(resolution=(1000, 1000))
shift = 24
skip_more = 10
for i in 1:1
    ii = (i - 1) ÷ 1 + 1
    jj = (i - 1) % 1 + 1
    ax = LScene(fig[ii, jj]; show_axis=false)
    skip = skips[i+shift]
    inds = 1:skip*skip_more:size(joined_timeseries)[2]
    xmax_ind = argmax(joined_timeseries[1, inds])
    koopman_mode = pf100_modes[i + shift][1:skip_more:end]
    koopman_mode .*= sign.(koopman_mode[xmax_ind])
    upper_quantile = quantile(koopman_mode, 0.97)
    scatter!(ax, joined_timeseries[:, inds], color=koopman_mode, colormap=:balance, markersize=5, colorrange = (-upper_quantile, upper_quantile))
    rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
end
save(figure_directory * "/p100_koopman_zoom.png", fig)

##
shift = 20
indchoices = 1:5
skip_more = 1
fig = Figure(resolution=(1000, 400))
ax = Axis(fig[1,1])
colors = [:red, :blue, :green, :orange, :purple]
for i in indchoices
    ii = (i - 1) ÷ 1 + 1
    jj = (i - 1) % 1 + 1
    skip = skips[i+shift]
    inds = 1:skip*skip_more:size(joined_timeseries)[2]
    xmax_ind = argmax(joined_timeseries[1, inds])
    koopman_mode = q_modes[i + shift][1000:3000]
    koopman_mode .*= sign.(q_modes[i + shift][xmax_ind])
    koopman_mode ./= maximum(abs.(koopman_mode))
    lines!(ax, koopman_mode, color = (colors[i], 0.5))
end

ax = Axis(fig[1,2])
for i in indchoices
    ii = (i - 1) ÷ 1 + 1
    jj = (i - 1) % 1 + 1
    skip = skips[i+shift]
    inds = 1:skip*skip_more:size(joined_timeseries)[2]
    xmax_ind = argmax(joined_timeseries[1, inds])
    koopman_mode = pf1_modes[i + shift][1000:3000]
    koopman_mode .*= sign.(q_modes[i + shift][xmax_ind])
    koopman_mode ./= maximum(abs.(koopman_mode))
    lines!(ax, koopman_mode, color = (colors[i], 0.5))
end


ax = Axis(fig[2, 1])
for i in indchoices
    ii = (i - 1) ÷ 1 + 1
    jj = (i - 1) % 1 + 1
    skip = skips[i+shift]
    inds = 1:skip*skip_more:size(joined_timeseries)[2]
    xmax_ind = argmax(joined_timeseries[1, inds])
    koopman_mode = pf10_modes[i + shift][1000:3000]
    koopman_mode .*= sign.(pf10_modes[i + shift][xmax_ind])
    koopman_mode ./= maximum(abs.(koopman_mode))
    lines!(ax, koopman_mode, color = (colors[i], 0.5))
end

ax = Axis(fig[2, 2])
for i in indchoices
    ii = (i - 1) ÷ 1 + 1
    jj = (i - 1) % 1 + 1
    skip = skips[i+shift]
    inds = 1:skip*skip_more:size(joined_timeseries)[2]
    xmax_ind = argmax(joined_timeseries[1, inds])
    koopman_mode = pf100_modes[i + shift][1000:3000]
    koopman_mode .*= sign.(pf100_modes[i + shift][xmax_ind])
    koopman_mode ./= maximum(abs.(koopman_mode))
    lines!(ax, koopman_mode, color = (colors[i], 0.5))
end
display(fig)

save(figure_directory * "/koopman_modes_in_time.png", fig)