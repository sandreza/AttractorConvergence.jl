using MarkovChainHammer, ProgressBars, LinearAlgebra

hfile = h5open(pwd() * data_directory  * "/embedding.hdf5", "r")
markov_chain = read(hfile["markov_chain"])
coarse_markov_chain = read(hfile["coarse_markov_chains"])
probability = read(hfile["probability"])
coarse_probabilities = read(hfile["coarse_probabilities"])
close(hfile)

# compute transition matrix (should use only half the data)
Ps = []
N = size(coarse_markov_chain)[1]
N2 = N ÷ 2
for i in ProgressBar(11:14)
    P1 = perron_frobenius(coarse_markov_chain[1:N2, i])
    P2 = perron_frobenius(coarse_markov_chain[N2+1:end, i])
    P = (P1 + P2)/2
    push!(Ps, P)
end

Ws = []
for i in ProgressBar(1:4)
    Λ, W = eigen(Ps[i]')
    push!(Ws, real.(W[:, end-3]))
end

##
set_theme!(backgroundcolor=:white)
inds = 1:10:size(joined_timeseries)[2]
xmax_ind = argmax(joined_timeseries[1, inds])
fig = Figure(resolution=(1000, 1000))
for i in 1:4
    ii = (i - 1) ÷ 2 + 1
    jj = (i - 1) % 2 + 1
    ax = LScene(fig[ii, jj]; show_axis=false)
    markov_indices = coarse_markov_chain[inds, i+10]
    koopman_mode = [Ws[i][markov_indices[j]] for j in eachindex(markov_indices)]
    koopman_mode .*= sign.(koopman_mode[xmax_ind])
    upper_quantile = quantile(koopman_mode, 0.9)
    scatter!(ax, joined_timeseries[:, inds], color=koopman_mode, colormap=:balance, markersize=5, colorrange = (-upper_quantile, upper_quantile))
    rotate_cam!(ax.scene, (0.0, -10.5, 0.0))
end
display(fig)