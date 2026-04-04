using HDF5, GLMakie, StatsBase, MarkovChainHammer

data_directory = "./data_ks/"

autocorrelations_gt  = []
autocorrelations_pf = []
autocorrelations_I = []
#ProgressBar(enumerate([x -> x[1], x -> mean(x .^2), x -> abs.(fft(x)[4])]))
hfile = h5open(data_directory * "autocorrelations_2.hdf5", "r")
push!(autocorrelations_gt, read(hfile["ground_truth 1"]))
push!(autocorrelations_gt, read(hfile["ground_truth 2"]))
push!(autocorrelations_pf, read(hfile["perron_frobenius 1"]))
push!(autocorrelations_pf, read(hfile["perron_frobenius 2"]))
push!(autocorrelations_I, read(hfile["irreversible 1"]))
push!(autocorrelations_I, read(hfile["irreversible 2"]))
close(hfile)

coarse_autocorrelations_gt = []
coarse_autocorrelations_pf = []
coarse_autocorrelations_I = []
hfile = h5open(data_directory * "/coarse_autocorrelations_2.hdf5", "r")
push!(coarse_autocorrelations_gt, read(hfile["ground_truth 1"]))
push!(coarse_autocorrelations_gt, read(hfile["ground_truth 2"]))
push!(coarse_autocorrelations_pf, read(hfile["perron_frobenius 1"]))
push!(coarse_autocorrelations_pf, read(hfile["perron_frobenius 2"]))
push!(coarse_autocorrelations_I, read(hfile["irreversible 1"]))
push!(coarse_autocorrelations_I, read(hfile["irreversible 2"]))
close(hfile)

# ProgressBar(enumerate([x -> real.(fft(x)[5]), x -> imag.(fft(x)[5]), x -> abs.(fft(x)[5]), x -> x[1] * x[2], x -> x[1] * x[32]]))
hfile = h5open(data_directory * "autocorrelations_3.hdf5", "r")
push!(autocorrelations_gt, read(hfile["ground_truth 1"])/64) 
push!(autocorrelations_pf, read(hfile["perron_frobenius 1"])/64) 
push!(autocorrelations_I, read(hfile["irreversible 1"])/64) 
# factor of 64 becuase of fft nonsense
close(hfile)

hfile = h5open(data_directory * "/coarse_autocorrelations_3.hdf5", "r")
push!(coarse_autocorrelations_gt, read(hfile["ground_truth 1"])/64)
push!(coarse_autocorrelations_pf, read(hfile["perron_frobenius 1"])/64)
push!(coarse_autocorrelations_I, read(hfile["irreversible 1"])/64)
close(hfile)

scale = 300
dt = 0.1
end_index = 500
ts = range(0, (end_index -1) * dt, length = end_index)
fig = Figure(resolution  = (3 * scale, 1 * scale))
ylabels = [L"\text{Autocorrelation of } u(0)",
           L"\text{Autocorrelation of } \frac{1}{L}\int_0^L u(x)^2 \, dx",
           L"\text{Autocorrelation of } \mathrm{Re}\,\hat{u}(4k)"]
for i in 1:3
    ax = Axis(fig[1, i], xlabel = L"\text{Time}", ylabel = ylabels[i])
    lines!(ax, ts, autocorrelations_gt[i][1:end_index], color = (:black, 0.5), label = "Ground Truth")
    lines!(ax,ts, autocorrelations_pf[i][1:end_index], color = (:red, 0.5), label = "Generator 10⁶")
    lines!(ax,ts, autocorrelations_I[i][1:end_index], color = (:blue, 0.5), label = "Irreversible 10⁶")
    lines!(ax, ts, coarse_autocorrelations_pf[i][1:end_index], color=(:red, 0.5), linestyle=:dash, label="Generator 10²")
    lines!(ax, ts, coarse_autocorrelations_I[i][1:end_index], color=(:blue, 0.5), linestyle=:dash, label="Irreversible 10²")

    if i == 3
        axislegend(ax, position = :rc)
    end
end
display(fig)
figure_directory = "./KSFigures"
save(figure_directory * "/temporal_autocorrelations.png", fig)