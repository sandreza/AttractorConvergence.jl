using HDF5
data_directory = "/real_data"
figure_directory = "unstructured_figures"
using GLMakie
hfile = h5open(pwd() * data_directory  * "/temporal_autocovariance.hdf5", "r")

time_autocovariance = read(hfile["time mean autocovariance"])
pf100_autocovariance_list = [read(hfile["ensemble mean autocovariance perron_frobenius 100 $i"]) for i in 8:25]
pf10_autocovariance_list = [read(hfile["ensemble mean autocovariance perron_frobenius 10 $i"]) for i in 8:25]
pf1_autocovariance_list = [read(hfile["ensemble mean autocovariance perron_frobenius 1 $i"]) for i in 8:25]
generator_autocovariance_list = [read(hfile["ensemble mean autocovariance generator $i"]) for i in 8:17]

close(hfile)


fig = Figure(resolution=(1000, 1000))
shift = 1
inds = 1:1001
timelist = (collect(inds) .-1 ) * .01
for i in 1:16
    ii = (i - 1) ÷ 4 + 1
    jj = (i - 1) % 4 + 1
    ax = Axis(fig[ii, jj])
    lines!(ax, timelist, time_autocovariance[inds], color=(:blue, 0.5), linewidth=4)
    lines!(ax, timelist, pf100_autocovariance_list[i + shift][inds], color=(:red, 0.5), linewidth=4)
end
display(fig)
save(figure_directory * "/pf100_autocorrelation.png", fig)

fig = Figure(resolution=(1000, 1000))
shift = 1
inds = 1:1001
timelist = (collect(inds) .-1 ) * .01
for i in 1:16
    ii = (i - 1) ÷ 4 + 1
    jj = (i - 1) % 4 + 1
    ax = Axis(fig[ii, jj])
    lines!(ax, timelist, time_autocovariance[inds], color=(:blue, 0.5), linewidth=4)
    lines!(ax, timelist, pf10_autocovariance_list[i + shift][1:10:end][inds], color=(:red, 0.5), linewidth=4)
end
display(fig)
save(figure_directory * "/pf10_autocorrelation.png", fig)

fig = Figure(resolution=(1000, 1000))
shift = 1
inds = 1:1001
timelist = (collect(inds) .-1 ) * .01
for i in 1:16
    ii = (i - 1) ÷ 4 + 1
    jj = (i - 1) % 4 + 1
    ax = Axis(fig[ii, jj])
    lines!(ax, timelist, time_autocovariance[inds], color=(:blue, 0.5), linewidth=4)
    lines!(ax, timelist, pf1_autocovariance_list[i + shift][1:100:end][inds], color=(:red, 0.5), linewidth=4)
end
display(fig)
save(figure_directory * "/pf1_autocorrelation.png", fig)

fig = Figure(resolution=(1000, 1000))
shift = 1
inds = 1:1001
timelist = (collect(inds) .-1 ) * .01
for i in 1:9
    ii = (i - 1) ÷ 3 + 1
    jj = (i - 1) % 3 + 1
    ax = Axis(fig[ii, jj])
    lines!(ax, timelist, time_autocovariance[inds], color=(:blue, 0.5), linewidth=4)
    lines!(ax, timelist, generator_autocovariance_list[i + shift][1:100:end][inds], color=(:red, 0.5), linewidth=4)
end
display(fig)
save(figure_directory * "/generator_autocorrelation.png", fig)

##
fig = Figure(resolution=(1000, 1000))
shift = 10
inds = 1:1001
timelist = (collect(inds) .-1 ) * .01
ax = Axis(fig[1, 1]; title = "generator")
lines!(ax, timelist, time_autocovariance[inds], color=(:blue, 0.5), linewidth=4)
lines!(ax, timelist, generator_autocovariance_list[shift][1:100:end][inds], color=(:red, 0.5), linewidth=4)
ax = Axis(fig[1, 2]; title = "perron-frobenius 1")
lines!(ax, timelist, time_autocovariance[inds], color=(:blue, 0.5), linewidth=4)
lines!(ax, timelist, pf1_autocovariance_list[shift][1:100:end][inds], color=(:red, 0.5), linewidth=4)
ax = Axis(fig[2, 1]; title = "perron-frobenius 10")
lines!(ax, timelist, time_autocovariance[inds], color=(:blue, 0.5), linewidth=4)
lines!(ax, timelist, pf10_autocovariance_list[shift][1:10:end][inds], color=(:red, 0.5), linewidth=4)
ax = Axis(fig[2, 2]; title = "perron-frobenius 100")
lines!(ax, timelist, time_autocovariance[inds], color=(:blue, 0.5), linewidth=4)
lines!(ax, timelist, pf100_autocovariance_list[shift][1:1:end][inds], color=(:red, 0.5), linewidth=4)
display(fig)
save(figure_directory * "/together_autocorrelation.png", fig)