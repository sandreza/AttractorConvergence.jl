using HDF5, SparseArrays, ProgressBars, AttractorConvergence, GLMakie
using LinearAlgebra, Statistics, FFTW, MarkovChainHammer, StateSpacePartitions
using StatsBase

function steady_state(partitions)
    nmax = maximum(partitions)
    p = zeros(nmax)
    count_vector = StatsBase.counts(partitions[:], 1:nmax)
    p = count_vector ./ length(partitions)
    return p ./ sum(p)
end



hfile = h5open(data_directory * "/ks.hdf5", "r")
timeseries = read(hfile["timeseries"])
close(hfile)

data_directory = "./data_ks"
hfile = h5open(data_directory * "/embedding.hdf5", "r")
partitions = read(hfile["coarse_markov_chains 9"])
close(hfile)

data_directory = "./data_ks"
hfile = h5open(data_directory * "/centers.hdf5", "r")
centers = read(hfile["centers 9"])
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

Q = (perron_frobenius_matrix - rperron_frobenius_matrix) / 2

hfile = h5open("data_ks/coarse_autocorrelations_2.hdf5", "w")
close(hfile)

observable(x) = x[1] # mean(x .^2)
for (i, observable) in ProgressBar(enumerate([x -> x[1], x -> mean(x .^ 2), x -> abs.(fft(x)[4])]))
    E = [observable(centers[:, i]) for i in 1:size(centers)[2]] # # [mean(centers[:, i] .^2) for i in 1:size(centers)[2]]
    Et = [observable(timeseries[1:2:128, i]) for i in ProgressBar(1:size(timeseries)[2])] # mean(timeseries .^2, dims = 1)[:]

    fig = Figure()
    ax = Axis(fig[1, 1])
    hist!(ax, Et, color=(:red, 0.5), normalization=:pdf, bins=100)
    hist!(ax, E, color=(:blue, 0.5), normalization=:pdf, bins=100)
    save("KSFigures/E_and_Et.png", fig)

    μET = mean(Et)
    μE = sum(E .* p)
    println("μET = $μET, μE = $μE", " μET/μE = $(μET/μE)")
    σET = std(Et)
    σE = sqrt(sum(E .^ 2 .* p) - μE^2)
    println("σET = $σET, σE = $σE", " σET/σE = $(σET/σE)")

    tac = real.(ifft(fft(Et) .* ifft(Et))) .- μET^2
    lines(tac[1:1000])

    F = E .* p
    autocor = zeros(1000)
    autocor[1] = E' * F
    for i in ProgressBar(2:1000)
        F .= (perron_frobenius_matrix * F)
        autocor[i] = E' * F
    end
    tmp = autocor .- μE^2

    lines(tac[1:1000])
    scatter!(tmp[1:1000])

    F = E .* p
    scale = 1
    autocor_I = zeros(scale * 1000)
    autocor_I[1] = E' * F
    dt = 1 / (2 * scale)
    # RK4
    for i in ProgressBar(2:scale*1000)
        # F .= F + dt * (perron_frobenius_matrix * F - p .* (perron_frobenius_matrix' * (F ./ p)))
        #=
        k1 = (perron_frobenius_matrix * F - p .* (perron_frobenius_matrix' * (F ./ p)))
        tmp2 = F + dt * k1 / 2
        k2 = (perron_frobenius_matrix * (tmp2) - p .* (perron_frobenius_matrix' * ((tmp2) ./ p)))
        tmp3 = F + dt * k2 / 2
        k3 = (perron_frobenius_matrix * (tmp3) - p .* (perron_frobenius_matrix' * ((tmp3) ./ p)))
        tmp4 = F + dt * k3
        k4 = (perron_frobenius_matrix * (tmp4) - p .* (perron_frobenius_matrix' * ((tmp4) ./ p)))
        =#
        k1 = (Q * F - p .* (Q' * (F ./ p)))
        tmp2 = F + dt * k1 / 2
        k2 = (Q * (tmp2) - p .* (Q' * ((tmp2) ./ p)))
        tmp3 = F + dt * k2 / 2
        k3 = (Q * (tmp3) - p .* (Q' * ((tmp3) ./ p)))
        tmp4 = F + dt * k3
        k4 = (Q * (tmp4) - p .* (Q' * ((tmp4) ./ p)))
        F .= F + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        autocor_I[i] = E' * F
    end
    tmp_I = autocor_I .- μE^2

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="time", ylabel="autocovariance")
    lines!(ax, tac[1:1000], color=(:black, 0.5), label="True")
    lines!(ax, tmp[1:1000], color=(:red, 0.5), label="Perron-Frobenius")
    lines!(ax, tmp_I[1:1000], color=(:blue, 0.5), label="Irreversible")
    axislegend(ax, position=:rt)
    save("KSFigures/coarse_autocorrelations_$i.png", fig)
    display(fig)

    hfile = h5open("data_ks/coarse_autocorrelations_2.hdf5", "r+")
    hfile["ground_truth $i"] = tac
    hfile["perron_frobenius $i"] = tmp
    hfile["irreversible $i"] = tmp_I
    close(hfile)
end


hfile = h5open("data_ks/coarse_autocorrelations_3.hdf5", "w")
close(hfile)

observable(x) = x[1] # mean(x .^2)
for (i, observable) in ProgressBar(enumerate([x -> real.(fft(x)[5]), x -> imag.(fft(x)[5]), x -> abs.(fft(x)[5]), x -> x[1] * x[2], x -> x[1] * x[32]]))
    E = [observable(centers[:, i]) for i in 1:size(centers)[2]] # # [mean(centers[:, i] .^2) for i in 1:size(centers)[2]]
    Et = [observable(timeseries[1:2:128, i]) for i in ProgressBar(1:size(timeseries)[2])] # mean(timeseries .^2, dims = 1)[:]

    fig = Figure()
    ax = Axis(fig[1, 1])
    hist!(ax, Et, color=(:red, 0.5), normalization=:pdf, bins=100)
    hist!(ax, E, color=(:blue, 0.5), normalization=:pdf, bins=100)
    save("KSFigures/E_and_Et.png", fig)

    μET = mean(Et)
    μE = sum(E .* p)
    println("μET = $μET, μE = $μE", " μET/μE = $(μET/μE)")
    σET = std(Et)
    σE = sqrt(sum(E .^ 2 .* p) - μE^2)
    println("σET = $σET, σE = $σE", " σET/σE = $(σET/σE)")

    tac = real.(ifft(fft(Et) .* ifft(Et))) .- μET^2
    lines(tac[1:1000])

    F = E .* p
    autocor = zeros(1000)
    autocor[1] = E' * F
    for i in ProgressBar(2:1000)
        F .= (perron_frobenius_matrix * F)
        autocor[i] = E' * F
    end
    tmp = autocor .- μE^2

    lines(tac[1:1000])
    scatter!(tmp[1:1000])

    F = E .* p
    scale = 1
    autocor_I = zeros(scale * 1000)
    autocor_I[1] = E' * F
    dt = 1 / (2 * scale)
    # RK4
    for i in ProgressBar(2:scale*1000)
        # F .= F + dt * (perron_frobenius_matrix * F - p .* (perron_frobenius_matrix' * (F ./ p)))
        #=
        k1 = (perron_frobenius_matrix * F - p .* (perron_frobenius_matrix' * (F ./ p)))
        tmp2 = F + dt * k1 / 2
        k2 = (perron_frobenius_matrix * (tmp2) - p .* (perron_frobenius_matrix' * ((tmp2) ./ p)))
        tmp3 = F + dt * k2 / 2
        k3 = (perron_frobenius_matrix * (tmp3) - p .* (perron_frobenius_matrix' * ((tmp3) ./ p)))
        tmp4 = F + dt * k3
        k4 = (perron_frobenius_matrix * (tmp4) - p .* (perron_frobenius_matrix' * ((tmp4) ./ p)))
        =#
        k1 = (Q * F - p .* (Q' * (F ./ p)))
        tmp2 = F + dt * k1 / 2
        k2 = (Q * (tmp2) - p .* (Q' * ((tmp2) ./ p)))
        tmp3 = F + dt * k2 / 2
        k3 = (Q * (tmp3) - p .* (Q' * ((tmp3) ./ p)))
        tmp4 = F + dt * k3
        k4 = (Q * (tmp4) - p .* (Q' * ((tmp4) ./ p)))
        F .= F + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        autocor_I[i] = E' * F
    end
    tmp_I = autocor_I .- μE^2

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="time", ylabel="autocovariance")
    lines!(ax, tac[1:1000], color=(:black, 0.5), label="True")
    lines!(ax, tmp[1:1000], color=(:red, 0.5), label="Perron-Frobenius")
    lines!(ax, tmp_I[1:1000], color=(:blue, 0.5), label="Irreversible")
    axislegend(ax, position=:rt)
    save("KSFigures/coarse_autocorrelations_3_$i.png", fig)
    display(fig)

    hfile = h5open("data_ks/coarse_autocorrelations_3.hdf5", "r+")
    hfile["ground_truth $i"] = tac
    hfile["perron_frobenius $i"] = tmp
    hfile["irreversible $i"] = tmp_I
    close(hfile)
end