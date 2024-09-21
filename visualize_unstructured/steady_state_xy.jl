using HDF5, ProgressBars, CairoMakie
# data_directory = "/nobackup1/sandre/AttractorConvergence/data/"
hfile = h5open(data_directory * "time_mean_statistics.hdf5", "r")
xmoments = read(hfile["x moments"])
xcumulants = read(hfile["x cumulants"])
ymoments = read(hfile["y moments"])
ycumulants = read(hfile["y cumulants"])
zmoments = read(hfile["z moments"])
zcumulants = read(hfile["z cumulants"])
close(hfile)

Npartitions = 25

hfile = h5open(data_directory * "centers.hdf5", "r")
centers = []
for i in 1:Npartitions
    center = read(hfile["centers $i"])
    push!(centers, center)
end
close(hfile)
hfile = h5open(data_directory * "eigenvalues.hdf5", "r")
probabilities = []
for i in 1:Npartitions
    p = read(hfile["generator steady state $i"])
    push!(probabilities, p)
end
close(hfile)

observables = [i -> i[1]^j for j in 1:5]
observables = vcat(observables, [i -> i[2]^j for j in 1:5])
@info "computing observables from model"

observables_list_model = zeros(Npartitions, length(observables))
for i in ProgressBar(1:Npartitions)
    p = probabilities[i]
    for j in eachindex(p)
        for k in eachindex(observables)
            observables_list_model[i, k] += observables[k](centers[i][:, j]) .* p[j]
        end
    end
end

cumulants_list_model = similar(observables_list_model)
cumulants_list_model[:, 1] = observables_list_model[:, 1]
cumulants_list_model[:, 2] = observables_list_model[:, 2] - observables_list_model[:, 1].^2
cumulants_list_model[:, 3] = observables_list_model[:, 3] - 3 * observables_list_model[:, 1] .* observables_list_model[:, 2] + 2 * observables_list_model[:, 1].^3
cumulants_list_model[:, 4] = observables_list_model[:, 4] - 4 * observables_list_model[:, 1] .* observables_list_model[:, 3] - 3 * observables_list_model[:, 2].^2 + 12 * observables_list_model[:, 1].^2 .* observables_list_model[:, 2] - 6 * observables_list_model[:, 1].^4
cumulants_list_model[:, 5] = observables_list_model[:, 5] - 5 * observables_list_model[:, 1] .* observables_list_model[:, 4] - 10 * observables_list_model[:, 2] .* observables_list_model[:, 3] + 20 * observables_list_model[:, 1].^2 .* observables_list_model[:, 3] + 30 * observables_list_model[:, 1] .* observables_list_model[:, 2].^2 - 60 * observables_list_model[:, 1].^3 .* observables_list_model[:, 2] + 24 * observables_list_model[:, 1].^5

cumulants_list_model[:, 1+5] = observables_list_model[:, 1+5]
cumulants_list_model[:, 2+5] = observables_list_model[:, 2+5] - observables_list_model[:, 1+5].^2
cumulants_list_model[:, 3+5] = observables_list_model[:, 3+5] - 3 * observables_list_model[:, 1+5] .* observables_list_model[:, 2+5] + 2 * observables_list_model[:, 1+5].^3
cumulants_list_model[:, 4+5] = observables_list_model[:, 4+5] - 4 * observables_list_model[:, 1+5] .* observables_list_model[:, 3+5] - 3 * observables_list_model[:, 2+5].^2 + 12 * observables_list_model[:, 1+5].^2 .* observables_list_model[:, 2+5] - 6 * observables_list_model[:, 1+5].^4
cumulants_list_model[:, 5+5] = observables_list_model[:, 5+5] - 5 * observables_list_model[:, 1+5] .* observables_list_model[:, 4+5] - 10 * observables_list_model[:, 2+5] .* observables_list_model[:, 3+5] + 20 * observables_list_model[:, 1+5].^2 .* observables_list_model[:, 3+5] + 30 * observables_list_model[:, 1+5] .* observables_list_model[:, 2+5].^2 - 60 * observables_list_model[:, 1+5].^3 .* observables_list_model[:, 2+5] + 24 * observables_list_model[:, 1+5].^5


ms = 20
ls = 40
lw = 5
labels = ["x κ₁", "x κ₂", " x κ₃", "x κ₄", "x κ₅", "y κ₁", "y κ₂", "y κ₃", "y κ₄", "y κ₅"]
xycumulants = vcat(xcumulants, ycumulants)
log10cumulantserror = log10.(abs.(cumulants_list_model .- reshape(xycumulants, (1, length(observables))))) .- log10.(abs.(reshape(xycumulants, (1, length(observables)))))
log10partition_numbers = log10.([size(centers[i])[2] for i in 1:Npartitions])
colors = [:red, :green, :blue, :orange, :purple, :red, :green, :blue, :orange, :purple]
colors[2] = :red 
colors[4] = :green
colors[7] = :blue
colors[9] = :orange
fig = Figure(resolution = (2000, 1500))
ax = Axis(fig[1, 1]; xlabel = "log10(partitions)", ylabel = "log10(relative error)", xlabelsize = ls, ylabelsize = ls, xticklabelsize = ls, yticklabelsize = ls)
for i in 2:2:5
    scatter!(ax, log10partition_numbers, log10cumulantserror[:, i], color = (colors[i]), markersize = 20, label = labels[i])
end
for i in 7:2:10
    scatter!(ax, log10partition_numbers, log10cumulantserror[:, i], color = (colors[i]), markersize = 20, label = labels[i])
end
axislegend(ax, position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)
lines!(ax, log10partition_numbers, - log10partition_numbers .+ 0.5, color = (:black, 0.5), linestyle=:dash, linewidth = lw, label = "-1 slope")


# figure_directory = pwd() * "/unstructured_figures"; figure_number = 4; save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)