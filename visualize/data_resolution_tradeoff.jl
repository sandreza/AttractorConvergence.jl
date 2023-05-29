

level = 9
eigenvalues_resolution_list = Vector{ComplexF64}[]
resolution_skip_list = [1, 10, 100]
timestep_end_list = div.(length(markov_chain), [1, 10, 100])
resolution_time_end_pair = []
for resolution_skip in ProgressBar(resolution_skip_list)
    for timestep_end in ProgressBar(timestep_end_list)
        markov_states = get_markov_states(centers_list, level)
        coarse_grained_markov_chain = div.(markov_chain .- 1, 2^(levels - level)) .+ 1
        coarse_grained_markov_chain_2 = div.(s_markov_chain .- 1, 2^(levels - level)) .+ 1
        Δt_c = Δt * resolution_skip
        inds_c = 1:resolution_skip:timestep_end
        prior = uninformative_prior(2^level)
        Q1 = BayesianGenerator(coarse_grained_markov_chain[inds_c], prior; dt=Δt_c)
        Q2 = BayesianGenerator(coarse_grained_markov_chain_2[inds_c], Q1.posterior; dt=Δt_c)
        Q = mean(Q2)
        Λ, V = eigen(Q)
        push!(eigenvalues_resolution_list, Λ)
        push!(resolution_time_end_pair, (resolution_skip, timestep_end))
    end
end

##
set_theme!(backgroundcolor=:white)
fig = Figure(resolution=(2000, 2000))
for ii in 1:9
    i = div(ii - 1, 3) + 1
    j = mod(ii - 1, 3) + 1
    start_value = ii
    rez = resolution_time_end_pair[ii][1]
    end_value = resolution_time_end_pair[ii][2]
    Δt_c = Δt * rez
    Tfinal = round(Int, end_value * Δt)
    ax = Axis(fig[j, i]; title="Δt⁻¹ =  $(round(Int, 1/Δt_c)), T = $(Tfinal), Partitions = $(2^9)", xlabel="real", ylabel="imaginary")
    eigenlist = eigenvalues_resolution_list[ii]
    scatter!(ax, real.(eigenlist), imag.(eigenlist), color=(:red, 0.5), label="level $(start_value)")
    ylims!(ax, (-70, 70))
    xlims!(ax, (-210, 10))
    # axislegend(ax)
end
display(fig)

##
level = 5
eigenvalues_resolution_list = Vector{ComplexF64}[]
resolution_time_end_pair = []
for resolution_skip in ProgressBar(resolution_skip_list)
    for timestep_end in ProgressBar(timestep_end_list)
        markov_states = get_markov_states(centers_list, level)
        coarse_grained_markov_chain = div.(markov_chain .- 1, 2^(levels - level)) .+ 1
        coarse_grained_markov_chain_2 = div.(s_markov_chain .- 1, 2^(levels - level)) .+ 1
        Δt_c = Δt * resolution_skip
        inds_c = 1:resolution_skip:timestep_end
        Q1 = BayesianGenerator(coarse_grained_markov_chain[inds_c]; dt=Δt_c)
        Q2 = BayesianGenerator(coarse_grained_markov_chain_2[inds_c], Q1.posterior; dt=Δt_c)
        Q = mean(Q2)
        Λ, V = eigen(Q)
        push!(eigenvalues_resolution_list, Λ)
        push!(resolution_time_end_pair, (resolution_skip, timestep_end))
    end
end

set_theme!(backgroundcolor=:white)
fig2 = Figure(resolution=(2000, 2000))
for ii in 1:9
    i = div(ii - 1, 3) + 1
    j = mod(ii - 1, 3) + 1
    start_value = ii
    rez = resolution_time_end_pair[ii][1]
    end_value = resolution_time_end_pair[ii][2]
    Δt_c = Δt * rez
    Tfinal = round(Int, end_value * Δt)
    ax = Axis(fig2[j, i]; title="Δt⁻¹ =  $(round(Int, 1/Δt_c)), T = $(Tfinal), Partitions = $(2^6)", xlabel="real", ylabel="imaginary")
    eigenlist = eigenvalues_resolution_list[ii]
    scatter!(ax, real.(eigenlist), imag.(eigenlist), color=(:blue, 0.5), label="level $(start_value)")
    ylims!(ax, (-30, 30))
    xlims!(ax, (-70, 10))
    # axislegend(ax)
end
display(fig2)