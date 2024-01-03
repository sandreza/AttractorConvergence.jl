struct RungeKutta4{S, T}
    k⃗::S
    x̃::T
    xⁿ⁺¹::T
end
RungeKutta4(n) = RungeKutta4(zeros(n,4), zeros(n), zeros(n))
function (step::RungeKutta4)(f, x, dt)
    @inbounds let
    @. step.x̃ = x
    step.k⃗[:, 1] .= f(step.x̃)
    @. step.x̃ = x + step.k⃗[:, 1] * dt / 2
    step.k⃗[:, 2] .= f(step.x̃)
    @. step.x̃ = x + step.k⃗[:, 2] * dt / 2
    step.k⃗[:, 3] .= f(step.x̃)
    @. step.x̃ = x + step.k⃗[:, 3] * dt
    step.k⃗[:, 4] .= f(step.x̃)
    @. step.xⁿ⁺¹ = x + (step.k⃗[:, 1] + 2 * step.k⃗[:, 2] + 2 * step.k⃗[:, 3] + step.k⃗[:, 4]) * dt / 6
    end
end

struct SparseGenerator{S, T}
    Q::S
    dt::T
end

(s::SparseGenerator)(p) = s.Q * p

function steady_state(partitions)
    p = zeros(maximum(partitions))
    for i in ProgressBar(eachindex(partitions))
        p[partitions[i]] += 1
    end
    return p ./ sum(p)
end

