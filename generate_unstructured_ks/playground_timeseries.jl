data_directory = "./data_ks"
hfile = h5open(data_directory * "/ks.hdf5", "r")
partitions = read(hfile["timeseries"])
close(hfile)


scale = 500
fig = Figure(resolution = (2*scale, 1 * scale))
index_range = 2000:6000
ts = collect(index_range)*dt
xs = collect(0:127)/128*34
ax = Axis(fig[1:2, 1], xlabel = L"\text{Space}", ylabel = L"\text{Time}", xlabelsize = 20, ylabelsize = 20)
heatmap!(ax, xs, ts, partitions[:, index_range], colormap = :balance, colorrange = (-3, 3))
ax = Axis(fig[1, 2], xlabel = L"\text{Space}", ylabel = L"\text{Amplitude}", xlabelsize = 20, ylabelsize = 20, title = "t = " * string(index_range[751]*dt))
lines!(ax, xs, partitions[:, index_range[750]], color = :blue)
ax = Axis(fig[2, 2], xlabel = L"\text{Space}", ylabel = L"\text{Amplitude}", xlabelsize = 20, ylabelsize = 20, title = "t = " * string(index_range[3001]*dt))
lines!(ax, xs, partitions[:, index_range[3001]], color = :red)
display(fig)
save("KSFigures/timeseries.png", fig)