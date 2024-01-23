using HDF5, Printf
data_directory = "/real_data"
figure_directory = "unstructured_figures"
using GLMakie
hfile = h5open(pwd() * data_directory  * "/temporal_autocovariance.hdf5", "r")
time_autocovariance = read(hfile["time mean autocovariance"])
pf100_autocovariance_list = [read(hfile["ensemble mean autocovariance perron_frobenius 100 $i"]) for i in 8:25]
pf10_autocovariance_list = [read(hfile["ensemble mean autocovariance perron_frobenius 10 $i"]) for i in 8:25]
pf1_autocovariance_list = [read(hfile["ensemble mean autocovariance perron_frobenius 1 $i"]) for i in 8:25]
generator_autocovariance_list = [read(hfile["ensemble mean autocovariance generator $i"]) for i in 8:25]
close(hfile)
e_hfile = h5open(pwd() * data_directory  * "/ensemble_mean_statistics.hdf5", "r")
partition_numbers = [read(e_hfile["number of partitions $i"]) for i in 8:25]
close(e_hfile)

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
for i in 1:16
    ii = (i - 1) ÷ 4 + 1
    jj = (i - 1) % 4 + 1
    ax = Axis(fig[ii, jj])
    lines!(ax, timelist, time_autocovariance[inds], color=(:blue, 0.5), linewidth=4)
    lines!(ax, timelist, generator_autocovariance_list[i + shift][1:100:end][inds], color=(:red, 0.5), linewidth=4)
end
display(fig)
save(figure_directory * "/generator_autocorrelation.png", fig)

##
fig = Figure(resolution=(1000, 1000))
shift = 17
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

##
fig = Figure(resolution=(1000, 1000))
shift = length(pf1_autocovariance_list) - 4
inds = 1:1001
timelist = (collect(inds) .-1 ) * .01
indchoices = [1, 7, 12, 18]
ls = 20
ylabels = [@sprintf("log10(N) = %2.2f", log10(partition_numbers[indchoice])) for indchoice in indchoices]
common_options = (; xlabel = "time", xlabelsize = ls, ylabelsize = ls, xticklabelsize = ls, yticklabelsize = ls, titlesize = ls)
for i in 1:4
    if i == 1
        ax = Axis(fig[i, 1]; ylabel = ylabels[i], title = "Generator", common_options...)
    else
        ax = Axis(fig[i, 1]; ylabel = ylabels[i], common_options...)
    end
    lines!(ax, timelist, time_autocovariance[inds], color=(:blue, 0.5), linewidth=4, label = "Time")
    lines!(ax, timelist, generator_autocovariance_list[indchoices[i]][1:100:end][inds], color=(:red, 0.5), linewidth=4, label = "Ensemble")
    if i == 1
        axislegend(ax, position=:rt, framecolor=(:grey, 0.5), patchsize=(ls, ls), labelsize=ls)
    end
    if i != 4
        hidexdecorations!(ax)
    end
end
for i in 1:4
    if i == 1
        ax = Axis(fig[i, 2]; title = "Perron-Frobenius 1", common_options...)
    else
        ax = Axis(fig[i, 2]; common_options...)
    end
    lines!(ax, timelist, time_autocovariance[inds], color=(:blue, 0.5), linewidth=4)
    lines!(ax, timelist, pf1_autocovariance_list[indchoices[i]][1:100:end][inds], color=(:red, 0.5), linewidth=4)
    if i != 4
        hidexdecorations!(ax)
    end
    hideydecorations!(ax)
end
for i in 1:4
    if i == 1
        ax = Axis(fig[i, 3]; title = "Perron-Frobenius 10", common_options...)
    else
        ax = Axis(fig[i, 3]; common_options...)
    end
    lines!(ax, timelist, time_autocovariance[inds], color=(:blue, 0.5), linewidth=4)
    lines!(ax, timelist, pf10_autocovariance_list[indchoices[i]][1:10:end][inds], color=(:red, 0.5), linewidth=4)
    if i != 4
        hidexdecorations!(ax)
    end
    hideydecorations!(ax)
end
for i in 1:4
    if i == 1
        ax = Axis(fig[i, 4]; title = "Perron-Frobenius 100", common_options...)
    else
        ax = Axis(fig[i, 4]; common_options...)
    end
    lines!(ax, timelist, time_autocovariance[inds], color=(:blue, 0.5), linewidth=4)
    lines!(ax, timelist, pf100_autocovariance_list[indchoices[i]][1:1:end][inds], color=(:red, 0.5), linewidth=4)
    if i != 4
        hidexdecorations!(ax)
    end
    hideydecorations!(ax)
end
display(fig)
save(figure_directory * "/together_autocorrelation_2.png", fig)