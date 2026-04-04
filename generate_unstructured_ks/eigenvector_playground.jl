using HDF5, SparseArrays, ProgressBars, AttractorConvergence, GLMakie, StatsBase
using LinearAlgebra, Statistics, FFTW, MarkovChainHammer, StateSpacePartitions

function steady_state(partitions)
    nmax = maximum(partitions)
    p = zeros(nmax)
    count_vector = StatsBase.counts(partitions[:], 1:nmax)
    p = count_vector ./ length(partitions)
    return p ./ sum(p)
end

function return_pf_rpf(index)
    data_directory = "./data_ks"
    hfile = h5open(data_directory * "/embedding.hdf5", "r")
    partitions = read(hfile["coarse_markov_chains $index"])
    close(hfile)

    data_directory = "./data_ks"
    hfile = h5open(data_directory * "/centers.hdf5", "r")
    centers = read(hfile["centers $index"])
    close(hfile)

    rpartitions = copy(partitions)
    for i in ProgressBar(1:size(partitions)[2])
        rpartitions[:, i] .= reverse(partitions[:, i])
    end

    p = steady_state(partitions[:])
    T, K = size(partitions)
    ns = maximum(partitions)
    # Sum sparse count matrices across columns; duplicate indices are summed inside sparse(...)
    count_matrix = sum(
        sparse(@view(partitions[2:T, j]),
            @view(partitions[1:T-1, j]),
            ones(Int64, T - 1),  # smaller V saves memory
            ns, ns)
        for j in ProgressBar(1:K)
    )

    exits = [length(count_matrix[:, i].nzval) for i in ProgressBar(1:size(count_matrix)[2])]

    column_sum = sum(count_matrix, dims=1)
    perron_frobenius_matrix = Float64.(count_matrix)
    normalization = sum(count_matrix, dims=1)
    @inbounds for i in ProgressBar(eachindex(normalization))
        for j in perron_frobenius_matrix[:, i].nzind
            perron_frobenius_matrix[j, i] /= normalization[i]
        end
    end

    # reverse
    rcount_matrix = sum(
        sparse(@view(rpartitions[2:T, j]),
            @view(rpartitions[1:T-1, j]),
            ones(Int64, T - 1),  # smaller V saves memory
            ns, ns)
        for j in ProgressBar(1:K)
    )

    rperron_frobenius_matrix = Float64.(rcount_matrix)
    rnormalization = sum(rcount_matrix, dims=1)
    @inbounds for i in ProgressBar(eachindex(rnormalization))
        for j in rperron_frobenius_matrix[:, i].nzind
            rperron_frobenius_matrix[j, i] /= rnormalization[i]
        end
    end

    return perron_frobenius_matrix, rperron_frobenius_matrix, p
end

function return_pf(index)
    data_directory = "./data_ks"
    hfile = h5open(data_directory * "/embedding.hdf5", "r")
    partitions = read(hfile["coarse_markov_chains $index"])
    close(hfile)

    data_directory = "./data_ks"
    hfile = h5open(data_directory * "/centers.hdf5", "r")
    centers = read(hfile["centers $index"])
    close(hfile)

    rpartitions = copy(partitions)
    for i in ProgressBar(1:size(partitions)[2])
        rpartitions[:, i] .= reverse(partitions[:, i])
    end

    p = steady_state(partitions[:])
    T, K = size(partitions)
    ns = maximum(partitions)
    # Sum sparse count matrices across columns; duplicate indices are summed inside sparse(...)
    count_matrix = sum(
        sparse(@view(partitions[2:T, j]),
            @view(partitions[1:T-1, j]),
            ones(Int64, T - 1),  # smaller V saves memory
            ns, ns)
        for j in ProgressBar(1:K)
    )

    exits = [length(count_matrix[:, i].nzval) for i in ProgressBar(1:size(count_matrix)[2])]

    column_sum = sum(count_matrix, dims=1)
    perron_frobenius_matrix = Float64.(count_matrix)
    normalization = sum(count_matrix, dims=1)
    @inbounds for i in ProgressBar(eachindex(normalization))
        for j in perron_frobenius_matrix[:, i].nzind
            perron_frobenius_matrix[j, i] /= normalization[i]
        end
    end

    return perron_frobenius_matrix, p
end

function get_Λi(index)
    dt = 0.1 
    pf, p = return_pf(index)
    Q = (pf - I)/dt
    M = Diagonal(p)
    M⁻¹ = Diagonal(1 ./p)
    Qi = (Q - M * Q' * M⁻¹) / 2
    Λi, Vi = eigen(Array(Qi))
    return Λi, Vi
end

dt = 0.1
# 10 is 263
# 16 is 8482
coarse_index = 16
pf,  p = return_pf(coarse_index)

Q = (pf - I)/dt
M = Diagonal(p)
M⁻¹ = Diagonal(1 ./p)
Qr = (Q + M * Q' * M⁻¹) / 2
Qi = (Q - M * Q' * M⁻¹) / 2

Λ, V = eigen(Array(Q))

Λi, Vi = eigen(Array(Qi))

data_directory = "./data_ks"
hfile = h5open(data_directory * "/centers.hdf5", "r")
centers = read(hfile["centers $coarse_index"])
close(hfile)

mode_index = size(centers, 2) - 1
koopman_mode = real.(sum([centers[:, i] .* V[i, mode_index] for i in 1:size(centers)[2]]))
koopman_mode_i = real.(sum([centers[:, i] .* Vi[i, mode_index] for i in 1:size(centers)[2]]))

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"\text{Spatial index}", ylabel = L"\text{Amplitude}")
lines!(ax, koopman_mode_i, label = L"\text{Koopman mode (imag)}")
lines!(ax, centers[:, 1], color = :red, label = L"\text{Center field}")
axislegend(ax, position = :rt)
display(fig)

density(imag.(Λi), npoints=1000)



data_directory = "./data_ks"
if !isfile(data_directory * "/imag_eigenvalues.hdf5")
    lambdas = [get_Λi(i) for i in [12, 17]]
    println("writing imag_eigenvalues.hdf5")
    hfile = h5open(data_directory * "/imag_eigenvalues.hdf5", "w")
    hfile["imag_eigenvalues 12"] = imag.(lambdas[1][1])
    hfile["imag_eigenvalues 17"] = imag.(lambdas[2][1])
    hfile["imag_eigenvectors 12"] = lambdas[1][2]
    hfile["imag_eigenvectors 17"] = lambdas[2][2]
    close(hfile)
end

scale = 40
fig = Figure(resolution = (scale * 12, scale * 9))
ax = Axis(fig[1, 1], xlabel = L"\mathrm{Im}(\lambda)", ylabel = L"\text{Density}", xlabelsize = 20, ylabelsize = 20)
density!(ax, imag.(lambdas[1][1]), npoints=10000, color=(:red, 0.5), label=string(length(lambdas[1][1])) * " cells" )
density!(ax, imag.(lambdas[2][1]), npoints=10000, color=(:blue, 0.5), label=string(length(lambdas[2][1])) * " cells")
axislegend(ax, position = :rt)
display(fig)

data_directory = "./data_ks"
hfile = h5open(data_directory * "/centers.hdf5", "r")
centers_12 = read(hfile["centers 12"])
centers_17 = read(hfile["centers 17"])
close(hfile)


koopman_index_12 = argmin(abs.(imag.(lambdas[1][1]) .- 1.0))
koopman_index_17 = argmin(abs.(imag.(lambdas[2][1]) .- 1.0))
koopman_mode_12 = real.(sum([centers_12[:, i] .* lambdas[1][2][i, koopman_index_12] for i in 1:size(centers_12)[2]]))
koopman_mode_17 = real.(sum([centers_17[:, i] .* lambdas[2][2][i, koopman_index_17] for i in 1:size(centers_17)[2]]))

fig = Figure()
xs = collect(0:63 )/ 64 * 34
ax = Axis(fig[1, 1], xlabel = L"\text{x}", ylabel = L"\text{Amplitude}")
lines!(ax,xs, circshift(koopman_mode_12, -21) , label=string(length(lambdas[1][1])) * " cells", color = :red)
lines!(ax,xs, koopman_mode_17 , color = :blue, label = string(length(lambdas[2][1])) * " cells")
axislegend(ax, position = :rt)
display(fig)


scale = 40
fig = Figure(resolution=(scale * 24, scale * 9))
ax = Axis(fig[1, 1], xlabel=L"\mathrm{Im}(\lambda)", ylabel=L"\text{Density}", xlabelsize=20, ylabelsize=20, title=L"\text{Density of eigenvalues}")
density!(ax, imag.(lambdas[1][1]), npoints=10000, color=(:red, 0.5), label=string(length(lambdas[1][1])) * " cells")
density!(ax, imag.(lambdas[2][1]), npoints=10000, color=(:blue, 0.5), label=string(length(lambdas[2][1])) * " cells")
axislegend(ax, position=:rt)
ax = Axis(fig[1, 2], xlabel=L"\text{x}", ylabel=L"\text{Amplitude}", title=L"\text{Koopman mode } \omega = 1.0", xlabelsize=20, ylabelsize=20)
lines!(ax, xs, circshift(koopman_mode_12, -21), label=string(length(lambdas[1][1])) * " cells", color=:red)
lines!(ax, xs, koopman_mode_17, color=:blue, label=string(length(lambdas[2][1])) * " cells")
axislegend(ax, position=:rt)
display(fig)
save("KSFigures/eigenvalue_density_and_koopman_mode.png", fig)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"\mathrm{Im}(\lambda)", ylabel = L"\text{PDF}")
hist!(ax, imag.(lambdas[1][1]), bins=100, color=(:red, 0.5), normalization=:pdf, label=string(length(lambdas[1][1])) * " cells")
hist!(ax, imag.(lambdas[2][1]), bins=100, color=(:blue, 0.5), normalization=:pdf, label=string(length(lambdas[2][1])) * " cells")
axislegend(ax, position = :rt)
display(fig)