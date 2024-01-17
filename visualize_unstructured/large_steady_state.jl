using HDF5, GLMakie
data_directory = "/real_data"
figure_directory = "/unstructured_figures"
@info "Loading data"
e_hfile = h5open(pwd() * data_directory  * "/ensemble_mean_statistics.hdf5", "r")
t_hfile = h5open(pwd() * data_directory  * "/time_mean_statistics.hdf5", "r")

ensemble_mean = zeros(25, 5)
partition_number = zeros(25)
error_list = zeros(25, 5)
relative_error = zeros(25, 5)
time_mean = read(t_hfile["z cumulants"])
for i in 1:25
    ensemble_mean[i, :] .= read(e_hfile["z cumulants $i"])
    partition_number[i] = read(e_hfile["number of partitions $i"])
    error_list[i, :] .= abs.(ensemble_mean[i, :] .- time_mean)
    relative_error[i, :] = error_list[i, :] ./ abs.(time_mean)
end
close(e_hfile)
close(t_hfile)

@info "done data"

@info "plotting"
##
lw = 4
ms = 15
ls = 25
options = (; xlabelsize = ls, ylabelsize = ls, xticklabelsize = ls, yticklabelsize = ls, xticks = 0:7, yticks = -8:0)
fig = Figure(resolution = (750, 550))
ax = Axis(fig[1,1]; xlabel = "log10(partition number)", ylabel = "log10(relative error)", options...)
inds = 3:25
x = log10.(partition_number[inds])
colors = [:red, :purple, :blue, :orange, :green]
for i in 1:4
    y = log10.(relative_error[inds, i])
    scatter!(ax, x, y, color = (colors[i], 0.5), markersize = ms, label = "cumulant $i")
end
lines!(ax, x, -0.0 .- x, color = (:black, 0.5), linestyle=:dash, linewidth = lw)
axislegend(ax, position=:rt, framecolor=(:grey, 0.5), patchsize=(ls, ls), markersize=ms, labelsize=ls)
display(fig)
save(pwd() * figure_directory * "/cumulants_relative_error.png", fig)

##
