using ParallelKMeans, HDF5, SparseArrays
using MarkovChainHammer, AttractorConvergence
using ProgressBars, LinearAlgebra, Statistics, Random
using MarkovChainHammer.BayesianMatrix
import MarkovChainHammer.TransitionMatrix: generator, holding_times, steady_state, perron_frobenius, entropy
import MarkovChainHammer.Utils: histogram, autocovariance
using StateSpacePartitions

# random seed for reproducibility
Random.seed!(12345)
tic = time()

timesteps = 10^6
m_timeseries, Δt = lorenz_data(timesteps=timesteps, Δt=1e-2, ϵ = 0.0)

@info "starting bisecting k-means"
p_min = 1.4 * 1 / 200
@info "computing embedding"
Nmax = 50 * round(Int, 1/ p_min)
skip_index = maximum([round(Int, size(m_timeseries)[2] / Nmax), 1])
tic = Base.time()
F, H, edge_information, parent_to_children, global_to_local, centers_list, CC, local_to_global = unstructured_tree(m_timeseries[:, 1:skip_index:end], p_min; threshold = 1.0);
toc2 = Base.time()
println("time for kmeans: ", toc2 - tic, " seconds")
# graph_edges should be called edge_information
embedding = UnstructuredTree(global_to_local, centers_list, parent_to_children)
partitions = zeros(Int64, size(m_timeseries)[2])
@info "computing partition trajectory"
tic2 = Base.time()
for i in ProgressBar(eachindex(partitions))
    @inbounds partitions[i] = embedding(m_timeseries[:, i])
end
toc = Base.time()
println("time for computing partition trajectory: ", toc - tic2, " seconds")



timesteps = 10^7
m_timeseries_2, Δt_2 = lorenz_data(timesteps=timesteps, Δt=1e-3, ϵ = 0.0)
@info "starting bisecting k-means"
partitions_2 = zeros(Int64, size(m_timeseries_2)[2])
@info "computing partition trajectory"
tic2 = Base.time()
for i in ProgressBar(eachindex(partitions_2))
    @inbounds partitions_2[i] = embedding(m_timeseries_2[:, i])
end
toc = Base.time()
println("time for computing partition trajectory: ", toc - tic2, " seconds")

pf_10 = perron_frobenius(partitions; step = 10)
pf_100 = perron_frobenius(partitions_2; step = 100)

pf_2_10 = perron_frobenius(partitions; step = 1)
pf_2_100 = perron_frobenius(partitions_2; step = 10)

norm(pf_10 - pf_100) / norm(pf_10)

lambda_10 = eigvals(pf_10)
lambda_100 = eigvals(pf_100)

lambda_2_10 = eigvals(pf_2_10)
lambda_2_100 = eigvals(pf_2_100)

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "real part", ylabel = "imaginary part", title = "PF dt = 1e-1")
scatter!(ax, real.(lambda_10), imag.(lambda_10), color = (:red, 0.5))
scatter!(ax, real.(lambda_100), imag.(lambda_100), color = (:blue, 0.5))
ax = Axis(fig[1, 2]; xlabel = "real part", ylabel = "imaginary part", title = "PF dt = 1e-2")
scatter!(ax, real.(lambda_2_10), imag.(lambda_2_10), color = (:red, 0.5))
scatter!(ax, real.(lambda_2_100), imag.(lambda_2_100), color = (:blue, 0.5))
display(fig)