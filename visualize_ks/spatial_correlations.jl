using HDF5, GLMakie

@info "loading data for kmeans"
hfile = h5open(data_directory * "/ks.hdf5", "r")
timeseries = read(hfile["timeseries"])
joined_timeseries = timeseries # hcat(timeseries, circshift(timeseries, (32, 0)), circshift(timeseries, (64, 0)), circshift(timeseries, (96, 0)))
close(hfile)

hfile = h5open(data_directory * "/centers.hdf5", "r")
centers_list = read(hfile["centers"])
coarsest_centers_list = read(hfile["centers 9"])
coarse_centers_list = read(hfile["centers 14"])
close(hfile)

hfile = h5open(data_directory * "/embedding.hdf5", "r")
markov_chain = read(hfile["markov_chain"])
coarsest_markov_chain = read(hfile["coarse_markov_chains 9"])
coarse_markov_chain = read(hfile["coarse_markov_chains 14"])
close(hfile)

Q = sparse_generator(markov_chain)
p = steady_state(markov_chain)

cQ = generator(coarse_markov_chain)
cH = decomposition(cQ).volume_preserving
cΛ, cV = eigen(cQ)
p = real.(cV[:, end] ./ sum(cV[:, end]))

ccQ = generator(coarsest_markov_chain)
ccΛ, ccV = eigen(ccQ)
ccp = real.(ccV[:, end] ./ sum(ccV[:, end]))
p_large = steady_state(markov_chain)

stseries = timeseries[:, 1:100:end]
sac = [mean(stseries .* circshift(stseries, (i, 0))) for i in 0:127]
sac_ens_c = [mean(sum(coarse_centers_list .* circshift(coarse_centers_list, (i, 0)) .* reshape(p, (1, size(coarse_centers_list)[2])), dims=2)) for i in 0:127]
sac_ens = [mean(sum(centers_list .* circshift(centers_list, (i, 0)) .* reshape(p_large, (1, size(centers_list)[2])), dims=2)) for i in 0:127]

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, 0:127, sac, color = :black)
scatter!(ax, 0:127, sac_ens_c, color = (:blue, 0.5))
scatter!(ax, 0:127, sac_ens, color = (:red, 0.5))
save("KSFigures/spatial_correlations.png", fig)

fig = Figure()
ax = Axis(fig[1, 1])
scatter!(ax, real.(ccΛ), imag.(ccΛ), color = (:red, 0.5))
scatter!(ax, real.(cΛ), imag.(cΛ), color = (:blue, 0.5))
save("KSFigures/spatial_correlations_coarsest.png", fig)

koopman_mode_c = [sum(reshape(real.(cV[:, end-i]), (1, size(coarse_centers_list)[2])) .* coarse_centers_list, dims = 2)[:] for i in 1:10]
koopman_mode_cc = sum(reshape(real.(ccV[:, end-1]), (1, size(coarsest_centers_list)[2])) .* coarsest_centers_list, dims = 2)[:]

fig = Figure()
ax = Axis(fig[1, 1])
scatter!(ax, koopman_mode_c / maximum(koopman_mode_c), color = (:blue, 0.5))
scatter!(ax, koopman_mode_cc / maximum(koopman_mode_cc), color = (:red, 0.5))
save("KSFigures/spatial_correlations_koopman.png", fig)

scatter(timeseries[:, 210] / maximum(timeseries[:, 210]))
scatter!(-circshift(koopman_mode_c[end], 16) / maximum(koopman_mode_c[end]))

function autocovariance_H(g⃗, Q::Eigen, p, timelist; progress=false)

    autocov = zeros(length(timelist))
    # Q  = V Λ V⁻¹
    Λ, V = Q
    v1 = V \ (p .* g⃗)
    w1 = g⃗' * V
    μ = sum(p .* g⃗)
    progress ? iter = ProgressBar(eachindex(timelist)) : iter = eachindex(timelist)
    for i in iter
        autocov[i] = real(w1 * (exp.(Λ .* timelist[i]) .* v1) - μ^2)
    end
    return autocov
end

last_ind = 400
g⃗ = mean(coarse_centers_list .^2, dims = 1)[:]
ac_ens = autocovariance(g⃗, eigen(cQ), 1:last_ind; progress = true)
ac_ens_H = autocovariance_H(g⃗, eigen(cH), p, 1:last_ind; progress = true)


g_timeseries = mean(timeseries .^2, dims = 1)[:]
ac_ts = autocovariance(g_timeseries; timesteps=last_ind, progress=true)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, 1:last_ind, ac_ts, color = (:green, 0.5))
lines!(ax, 1:last_ind, ac_ens, color = (:blue, 0.5))
lines!(ax, 1:last_ind, ac_ens_H, color = (:red, 0.5))
save("KSFigures/spatial_correlations_autocovariance.png", fig)



#=
sQ = SparseGenerator(Q', dt);
rk4 = RungeKutta4(length(p))
observable_trajectory = copy(𝒪)
q_runge_kutta_correlation[1] = sum(𝒪 .* p .* observable_trajectory) .- sum(p .* 𝒪)^2
@info "generator runge-kutta autocovariance"
for jj in ProgressBar(2:numsteps)
    rk4(sQ, observable_trajectory, dt)
    observable_trajectory .= rk4.xⁿ⁺¹
    q_runge_kutta_correlation[jj] = sum(𝒪 .* p .* observable_trajectory) .- sum(p .* 𝒪)^2
end
=#

##
fig = Figure()
ax = Axis(fig[1, 1])
heatmap!(ax, timeseries[:, 1:1000])
ax = Axis(fig[1, 2])
scatter!(ax, timeseries[:, 1])
ax = Axis(fig[1, 3])
scatter!(ax, timeseries[:, 200])
ax = Axis(fig[1, 4])
scatter!(ax, timeseries[1, 1:1000])
display(fig)
##

scatter(timeseries[1, 1:1000])
##
new_centers = [mean(timeseries[:, markov_chain.==i], dims=2)[:] for i in ProgressBar(1:maximum(markov_chain))]
newer_centers = [timeseries[:, markov_chain.==i][:, 1] for i in ProgressBar(sort(union(markov_chain)))]
# vector to array 
new_centers_array = zeros(128, maximum(markov_chain))
for i in ProgressBar(1:length(new_centers))
    new_centers_array[:, i] .= new_centers[i]
end
newer_centers_array = zeros(128, length(newer_centers))
for i in ProgressBar(1:length(newer_centers))
    newer_centers_array[:, i] .= newer_centers[i]
end

nt = new_centers_array[(!).(isnan.(new_centers_array))]
hist(newer_centers_array[:], bins=100, normalization=:pdf)
hist!(timeseries[:, 1:100:end][:], bins = 100, normalization = :pdf)

centers_difference = [norm(new_centers[i] - centers_list[:, i]) for i in ProgressBar(1:length(new_centers))]

#
tseries_1 = timeseries[:, markov_chain.==1]
fig = Figure()
ax = Axis(fig[1, 1])
for i in 1:size(tseries_1)[2]
    lines!(ax, tseries_1[:, i], color = (:red, 0.05))
end
display(fig)
##