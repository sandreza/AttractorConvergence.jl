@info "evolving lorenz equations"
function lorenz_data(; timesteps=10^7, Δt=0.005, ϵ=0.0, ρ=t -> 28.0, initial_condition=[1.4237717232359446, 1.778970017190979, 16.738782836244038])
    rhs(x, t) = lorenz(x, ρ(t), 10.0, 8.0 / 3.0)
    x_f = zeros(3, timesteps)
    x_f[:, 1] .= initial_condition
    evolve! = RungeKutta4(3)
    for i in ProgressBar(2:timesteps)
        xOld = x_f[:, i-1]
        evolve!(rhs, xOld, Δt)
        if ϵ > 0.0
            𝒩 = randn(3)
            @inbounds @. x_f[:, i] = evolve!.xⁿ⁺¹ + ϵ * sqrt(Δt) * 𝒩
        else
            @inbounds @. x_f[:, i] = evolve!.xⁿ⁺¹
        end
    end
    return x_f, Δt
end
function lorenz_symmetry(timeseries)
    symmetrized_timeseries = zeros(size(timeseries))
    for i in ProgressBar(1:size(timeseries)[2])
        symmetrized_timeseries[1, i] = -timeseries[1, i]
        symmetrized_timeseries[2, i] = -timeseries[2, i]
        symmetrized_timeseries[3, i] = timeseries[3, i]
    end
    return symmetrized_timeseries
end

function distance_matrix(data)
    d_mat = zeros(size(data)[2], size(data)[2])
    for j in ProgressBar(1:size(data)[2])
        Threads.@threads for i in 1:j-1
            @inbounds d_mat[i, j] = norm(data[:, i] - data[:, j])
        end
    end
    return Symmetric(d_mat)
end
##
timesteps = 4*10^6
timeseries, Δt = lorenz_data(timesteps=timesteps, Δt=0.0025)
s_timeseries = lorenz_symmetry(timeseries)
joined_timeseries = hcat(timeseries, s_timeseries) # only for Partitioning Purpose
##
@info "saving data"
hfile = h5open(pwd() * "/data/lorenz.hdf5", "w")
hfile["timeseries"] = timeseries
hfile["symmetrized timeseries"] = s_timeseries
hfile["dt"] = Δt
close(hfile)