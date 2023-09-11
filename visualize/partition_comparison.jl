hfile = h5open("data/lorenz.hdf5", "r")
m_timeseries = read(hfile["timeseries"])
close(hfile)
hfile = h5open("data/structured_embedding.hdf5", "r")
fp_ab = read(hfile["markov_chain_intersection"])
ulam = read(hfile["markov_chain_ulam"])
close(hfile)
hfile = h5open("data/embedding.hdf5", "r")
markov_chain = read(hfile["markov_chain"])
close(hfile)
##
level = 9 # could automate, but hard coding 512ish for plotting and comparison
binary_tree = div.(markov_chain .+ 2^levels .- 1, 2^(levels - level)) .- minimum(div.(markov_chain .+ 2^levels .- 1, 2^(levels - level))) .+1
##
Q_binary_tree = mean(BayesianGenerator(binary_tree; dt = Δt))
Q_fp_ab = mean(BayesianGenerator(fp_ab; dt = Δt))
Q_ulam = mean(BayesianGenerator(ulam; dt = Δt))
##
Λ_binary_tree, V_binary_tree =  eigen(Q_binary_tree)
Λ_fp_ab, V_fp_ab =  eigen(Q_fp_ab)
Λ_ulam, V_ulam =  eigen(Q_ulam)
Λ_binary_tree[end-4: end-1]
Λ_fp_ab[end-4: end-1]
Λ_ulam[end-4: end-1]
##
p_binary_tree = real.(V_binary_tree[:, end] ./ sum(V_binary_tree[:, end]))
p_fp_ab = real.(V_fp_ab[:, end] ./ sum(V_fp_ab[:, end]))
p_ulam = real.(V_ulam[:, end] ./ sum(V_ulam[:, end]))
##
println("probability ratio for binary tree partition is ", maximum(p_binary_tree) / minimum(p_binary_tree))
println("probability ratio for coherent structure partition is ", maximum(p_fp_ab) / minimum(p_fp_ab))
println("probability ratio for ulam's method is ", maximum(p_ulam) / minimum(p_ulam))
##
marker_size = 10
set_theme!(backgroundcolor=:white)
fig = Figure(resolution = (3000, 2000))
ax11 = LScene(fig[1, 1]; show_axis = false)
scatter!(ax11, m_timeseries[:, inds]; color=binary_tree[inds], colormap=:glasbey_hv_n256, markersize=marker_size)
ax12 = LScene(fig[1, 2]; show_axis = false)
scatter!(ax12, m_timeseries[:, inds]; color=fp_ab[inds], colormap=:glasbey_hv_n256, markersize=marker_size)
ax13 = LScene(fig[1, 3]; show_axis = false)
scatter!(ax13, m_timeseries[:, inds]; color=ulam[inds], colormap=:glasbey_hv_n256, markersize=marker_size)

rotate_cam!(ax11.scene, (0.0, -10.5, 0.0))
rotate_cam!(ax12.scene, (0.0, -10.5, 0.0))
rotate_cam!(ax13.scene, (0.0, -10.5, 0.0))

#=
# if showing koopman mode, for visualization purposes can decided opacity of a cell by the probability of being in that cell
ax32 = LScene(fig[3, 2]; show_axis=false)
scatter!(ax32, m_timeseries[:, 1:10:length(memb_fp_km)]; color=colors_fp_km[1:10:length(memb_fp_km)], colormap=:balance, markersize=5)
=#
# rotate_cam!(ax11.scene, (0.0, -10.5, 0.0))
# rotate_cam!(ax32.scene, (0.0, -10.5, 0.0))
# rotate_cam!(ax13.scene, (0.0, -10.5, 0.0))

ax21 = Axis(fig[2, 1]; title = "binary tree means")
scatter!(ax21, real.(eigenvalues_list[level]), imag.(eigenvalues_list[level]), color = :red)
xlimsv = (-1010, 10)
ylimsv = (-110, 110)
xlims!(ax21, xlimsv)
ylims!(ax21, ylimsv)
ax22 = Axis(fig[2, 2]; title = "coherent structures")
scatter!(ax22, real.(Λ_fp_ab), imag.(Λ_fp_ab), color = :blue)
xlims!(ax22, xlimsv)
ylims!(ax22, ylimsv)
ax22 = Axis(fig[2, 3]; title = "Ulam's Method")
scatter!(ax22, real.(Λ_ulam), imag.(Λ_ulam), color = :green)
xlims!(ax22, xlimsv)
ylims!(ax22, ylimsv)
display(fig)

##
# note that the stingray tail is a sign of wasted resolution. I way around this is to use an embedding that tries to get a uniform partition
