hfile = h5open(pwd() * "/data/lorenz.hdf5", "r")
m_timeseries = read(hfile["timeseries"])
close(hfile)

@info "ab periodic orbit partition"
T = 1.5586522107162 # period
ic = [-1.3763610682134e1, -1.9578751942452e1, 27.0] # initial condition from Viswanath
subdiv = 2^11
dt_ab = T / subdiv
ab_periodic_state, dt  = lorenz_data(; timesteps=subdiv, Δt=dt_ab, ϵ=0.0, ρ=t -> 28.0, initial_condition=ic)
n_markov_states = 32 
skip = round(Int, subdiv / n_markov_states)
markov_states = [ab_periodic_state[:, i] for i in 1:skip:subdiv]
numstates = length(markov_states)

m_emb_ab = Int64[]
n = length(m_timeseries[1,:])
for j in ProgressBar(1:n)
    mi = argmin(norm.([m_timeseries[:,j] .- markov_states[i] for i in 1:numstates])) # could be a bit greedier here
    push!(m_emb_ab, mi)
end
##
@info "non-zero fixed point partition"
markov_states = fixed_points = [-sqrt(72), -sqrt(72), 27], [sqrt(72), sqrt(72), 27]# [[-sqrt(72), -sqrt(72), 27], [0.0, 0.0, 0.0], [sqrt(72), sqrt(72), 27]]
m_emb_fp_d = Vector{Float64}[]
n = length(m_timeseries[1,:])
@info "computing distances"
for j in ProgressBar(1:n)
    md = norm.([m_timeseries[:,j] .- markov_state for markov_state in markov_states])
    push!(m_emb_fp_d, md)
end
##
@info "Determine which fixed point each point on the attractor is closest to"
ms = argmin.(m_emb_fp_d)
distances_per_ms = [m_emb_fp_d[i][ms[i]] for i in eachindex(ms)]
ms1 = Float64[]
ms2 = Float64[]
ms3 = Float64[]
ms_fp = Float64[]
for i in ProgressBar(eachindex(m_emb_fp_d))
    if ms[i] == 1
        push!(ms1, distances_per_ms[i])
        push!(ms_fp, minimum(distances_per_ms[i]))
    elseif ms[i] == 2
        push!(ms2, distances_per_ms[i])
        push!(ms_fp, minimum(distances_per_ms[i]))
    elseif ms[i] == 3
        push!(ms3, distances_per_ms[i])
        push!(ms_fp, minimum(distances_per_ms[i]))
    end
end
##
@info "define markov embedding for fixed point partition"
N = 16
Δq = 1 / N # N * Δq = 1 for some N
quantile_list = Δq:Δq:1-Δq
threshold1 = quantile.(Ref(ms1), quantile_list)
threshold2 = quantile.(Ref(ms2), quantile_list)
# threshold3 = quantile.(Ref(ms3), quantile_list)
m_emb_fp = Int64[]
for i in ProgressBar(eachindex(ms))
    if ms[i] == 1
       mi = sum(distances_per_ms[i] .> threshold1) .+ 1
       push!(m_emb_fp, mi)
    elseif ms[i] == 2
        mi = sum(distances_per_ms[i] .> threshold2) .+ 1
        push!(m_emb_fp, mi + length(quantile_list)+1)
    elseif ms[i] == 3
        mi = sum(distances_per_ms[i] .> threshold3) .+ 1
        push!(m_emb_fp, mi + 2*(length(quantile_list)+1))
    end
end
##
@info "intersection of previous two partitions"
combined_list = Tuple{Int64, Int64}[]
for i in ProgressBar(eachindex(m_emb_fp))
    push!(combined_list, (m_emb_fp[i], m_emb_ab[i]))
end
base_states = sort(union(combined_list))
reverse_map = [Int64[] for i in 1:length(union(m_emb_fp))]
for i in ProgressBar(eachindex(base_states))
    push!(reverse_map[base_states[i][1]], base_states[i][2])
end
index_jump = length.(reverse_map)
index_jump = [0, cumsum(index_jump)[1:end-1]...]
##
m_emb_fp_ab = Int64[]
for i in ProgressBar(eachindex(combined_list))
    min_ind = argmin(abs.(reverse_map[combined_list[i][1]] .- combined_list[i][2]))
    push!(m_emb_fp_ab, min_ind + index_jump[combined_list[i][1]] )
end
##
@info "Ulam's method"
xmin, xmax = extrema(m_timeseries[1, :])
ymin, ymax = extrema(m_timeseries[2, :])
zmin, zmax = extrema(m_timeseries[3, :])
##
cells = 15
xgrid = collect(range(xmin, stop=xmax, length=cells+2))[2:end-1]
ygrid = collect(range(ymin, stop=ymax, length=cells+2))[2:end-1]
zgrid = collect(range(zmin, stop=zmax, length=cells+2))[2:end-1]

tuple_list = Tuple{Int64, Int64, Int64}[]
@info "tensor product grid"
for i in ProgressBar(1:size(m_timeseries,2))
    s⃗ = m_timeseries[:, i]
    ind1 = sum(s⃗[1] .< xgrid)
    ind2 = sum(s⃗[2] .< ygrid)
    ind3 = sum(s⃗[3] .< zgrid)
    push!(tuple_list, Tuple([ind1, ind2, ind3]))
end
@info "redundant states removed"
markov_state = union(tuple_list)
##
markov_index = Int64[]
for i in ProgressBar(eachindex(tuple_list))
    push!(markov_index, argmax([tuple_list[i]] .== markov_state))
    #=
    tl = tuple_list[i]
    @inbounds for j in eachindex(markov_state)
        ms = markov_state[j]
        if tl[1] == ms[1]
            if tl[2] == ms[2]
                if tl[3] == ms[3]
                    push!(markov_index, j)
                    break
                end
            end
        end
    end
    =#
end

##
@info "saving"
hfile = h5open(pwd() * "/data/structured_embedding.hdf5", "w")
hfile["markov_chain_ab_orbit"] = m_emb_ab
hfile["markov_chain_fixed_point"] = m_emb_fp
hfile["markov_chain_intersection"] = m_emb_fp_ab
hfile["markov_chain_ulam"] = markov_index
hfile["dt"] = dt
close(hfile)


#=
fig = Figure() 
ax1 = LScene(fig[1, 1]; show_axis=false)
scatter!(ax1, m_timeseries[:, 1:10:length(m_emb_fp)]; color=m_emb_fp[1:10:end], colormap=:glasbey_hv_n256, markersize=5)
ax2 = LScene(fig[1, 2]; show_axis=false)
scatter!(ax2, m_timeseries[:, 1:10:length(m_emb_fp)]; color=m_emb_ab[1:10:end], colormap=:glasbey_hv_n256, markersize=5)
ax3 = LScene(fig[1, 3]; show_axis=false)
scatter!(ax3, m_timeseries[:, 1:10:length(m_emb_fp)]; color=memb_fp_ab[1:10:end], colormap=:glasbey_hv_n256, markersize=5)
rotate_cam!(ax1.scene, (0.0, -10.5, 0.0))
rotate_cam!(ax2.scene, (0.0, -10.5, 0.0))
rotate_cam!(ax3.scene, (0.0, -10.5, 0.0))
display(fig)
=#