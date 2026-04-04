using HDF5, SparseArrays, ProgressBars, AttractorConvergence, GLMakie
using LinearAlgebra, Statistics, FFTW, MarkovChainHammer, StateSpacePartitions

function steady_state(partitions)
    p = zeros(maximum(partitions))
    for i in ProgressBar(eachindex(partitions))
        p[partitions[i]] += 1
    end
    return p ./ sum(p)
end


data_directory = "./data"
hfile = h5open(data_directory * "/embedding_32.hdf5", "r")
partitions = read(hfile["markov_chain"])
close(hfile)

hfile = h5open(data_directory * "/lorenz.hdf5", "r")
timeseries = read(hfile["timeseries"])
close(hfile)

data_directory = "./data"
hfile = h5open(data_directory * "/centers.hdf5", "r")
centers = read(hfile["centers"])
close(hfile)

rpartitions = reverse(partitions)

p = steady_state(partitions[:])
T = size(partitions)[1]
ns = maximum(partitions)
r_shaped_partitions = reshape(partitions, (T÷200, 200))
T200 = T÷200
# Sum sparse count matrices across columns; duplicate indices are summed inside sparse(...)
tic = time()
count_matrix = sum(sparse(@view(r_shaped_partitions[2:T200, j]), @view(r_shaped_partitions[1:T200-1, j]),ones(Int32, T200 - 1)) for j in ProgressBar(1:size(r_shaped_partitions)[2]))  # smaller V saves memoryns, ns)
toc = time()
println("Time taken to create count matrix: $(toc - tic) seconds")
column_sum = sum(count_matrix, dims=1)
perron_frobenius_matrix = Float64.(count_matrix)
normalization = sum(count_matrix, dims=1)
@inbounds for i in ProgressBar(eachindex(normalization))
    for j in perron_frobenius_matrix[:, i].nzind
        perron_frobenius_matrix[j, i] /= normalization[i]
    end
end

# reverse
r_shaped_partitions = reshape(rpartitions, (T÷200, 200))
T200 = T÷200
tic = time()
rcount_matrix = sum(sparse(@view(r_shaped_partitions[2:T200, j]), @view(r_shaped_partitions[1:T200-1, j]),ones(Int32, T200 - 1)) for j in ProgressBar(1:size(r_shaped_partitions)[2]))  # smaller V saves memoryns, ns)
toc = time()
println("Time taken to create reverse count matrix: $(toc - tic) seconds")


rperron_frobenius_matrix = Float64.(rcount_matrix)
rnormalization = sum(rcount_matrix, dims=1)
@inbounds for i in ProgressBar(eachindex(rnormalization))
    for j in rperron_frobenius_matrix[:, i].nzind
        rperron_frobenius_matrix[j, i] /= rnormalization[i]
    end
end

Q = (perron_frobenius_matrix - rperron_frobenius_matrix) / 2


observable(x) = x[3] # mean(x .^2)
E = [observable(centers[:, i]) for i in 1:size(centers)[2]] # # [mean(centers[:, i] .^2) for i in 1:size(centers)[2]]
Et = [observable(timeseries[:, i]) for i in ProgressBar(1:size(timeseries)[2])] # mean(timeseries .^2, dims = 1)[:]



μET = mean(Et)
μE = sum(E .* p)
println("μET = $μET, μE = $μE", " μET/μE = $(μET/μE)")
σET = std(Et)
σE = sqrt(sum(E .^ 2 .* p) - μE^2)
println("σET = $σET, σE = $σE", " σET/σE = $(σET/σE)")

tac = real.(ifft(fft(Et) .* ifft(Et))) .- μET^2
lines(tac[1:3000])

F = E .* p
autocor = zeros(40000)
autocor[1] = E' * F
for i in ProgressBar(2:40000)
    F .= (perron_frobenius_matrix * F)
    autocor[i] = E' * F
end
tmp = autocor .- μE^2

lines(tac[1:1:40000], color = (:black, 0.5))
lines!(tmp[1:40000], color = (:red, 0.5))

F = E .* p
scale = 1
autocor_I = zeros(scale * 40000)
autocor_I[1] = E' * F
dt = 1 / (2 * scale)
# RK4
for i in ProgressBar(2:scale*40000)
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
lines!(ax, tac[1:1:40000], color=(:black, 0.5), label="True")
lines!(ax, tmp[1:40000], color=(:red, 0.5), label="Perron-Frobenius")
lines!(ax, tmp_I[1:40000], color=(:blue, 0.5), label="Irreversible")
axislegend(ax, position=:rt)
display(fig)

