using HDF5, LinearAlgebra, Statistics, ProgressBars, MarkovChainHammer

# Lorenz system parameters
r = 28.0
sigma = 10.0
beta = 8/3

# Fixed points of Lorenz system
# Origin
fp_origin = [0.0, 0.0, 0.0]

# Left lobe: (-sqrt(beta*(r-1)), -sqrt(beta*(r-1)), r-1)
sqrt_beta_r_minus_1 = sqrt(beta * (r - 1))
fp_left = [-sqrt_beta_r_minus_1, -sqrt_beta_r_minus_1, r - 1]

# Right lobe: (sqrt(beta*(r-1)), sqrt(beta*(r-1)), r-1)
fp_right = [sqrt_beta_r_minus_1, sqrt_beta_r_minus_1, r - 1]

@info "Loading Lorenz data"
# Use path relative to project root (go up one directory from rbf_to_ulam/)
data_directory = joinpath(@__DIR__, "..", "data")
data_directory = abspath(data_directory)  # Normalize the path
hfile = h5open(joinpath(data_directory, "lorenz.hdf5"), "r")
timeseries = read(hfile["timeseries"])
dt = read(hfile["dt"])
close(hfile)

@info "Computing distances to fixed points"
N = size(timeseries, 2)
distances_origin = zeros(N)
distances_left = zeros(N)
distances_right = zeros(N)

for i in ProgressBar(1:N)
    state = timeseries[:, i]
    distances_origin[i] = norm(state - fp_origin)
    distances_left[i] = norm(state - fp_left)
    distances_right[i] = norm(state - fp_right)
end

# Stack distances into a matrix: each column is [d_origin, d_left, d_right] for one time point
distance_matrix = hcat(distances_left, distances_origin, distances_right)'

@info "Constructing Ulam's method dictionary"
# For Ulam's method, assign each state to the nearest fixed point
# left lobe = 1, origin = 2, right lobe = 3
ulam_markov_chain = zeros(Int, N)

for i in ProgressBar(1:N)
    d_origin = distances_origin[i]
    d_left = distances_left[i]
    d_right = distances_right[i]
    
    # Find which fixed point is closest
    distances = [d_left, d_origin, d_right]
    min_idx = argmin(distances)
    
    # Assign state: left lobe = 1, origin = 2, right lobe = 3
    ulam_markov_chain[i] = min_idx
end

@info "Creating one-hot encoding of Ulam's method states"
# One-hot encoding: each row is a time point, each column is a state (left lobe, origin, right lobe)
ulam_onehot = zeros(Float64, 3, N)
for i in 1:N
    ulam_onehot[ulam_markov_chain[i], i] = 1.0
end

@info "Constructing Ulam's method generator"
P_ulam = perron_frobenius(ulam_markov_chain; step = 1)
# Construct generator: Q = (P - I) / dt
I_ulam = Matrix{Float64}(LinearAlgebra.I, size(P_ulam))
Q_ulam = (P_ulam - I_ulam) / dt

@info "Constructing RBF dictionaries"
# RBF parameters: d = 10^{-alpha} for alpha in {-2, -1, 0, 1, 2}
alphas = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
d_values = [10.0^(-alpha) for alpha in alphas]

# For RBF, we use the distance matrix directly
# Each row corresponds to a fixed point, each column to a time point
# We'll create RBF features: exp(-distance^2 / (2*d^2))

rbf_operators = Dict{Int, Matrix{Float64}}()
rbf_generators = Dict{Int, Matrix{Float64}}()
rbf_eigenvalues = Dict{Int, Vector{ComplexF64}}()
rbf_eigenvectors = Dict{Int, Matrix{ComplexF64}}()

for (idx, d) in enumerate(d_values)
    alpha = alphas[idx]
    @info "Computing RBF dictionary for alpha = $alpha (d = $d)"
    
    # Compute RBF features: exp(-distance^2 / (2*d^2))
    # Shape: (3, N) where 3 is number of fixed points
    # Numerically stable: subtract minimum distance per column before exponentiating
    # This prevents underflow when distances are large relative to d
    min_distances = minimum(distance_matrix, dims=1)  # Shape: (1, N) - minimum per column
    
    # Subtract minimum from each column before exponentiating
    # This ensures the largest exponential is exp(0) = 1, preventing underflow
    squared_distance_matrix_shifted = @. (distance_matrix^2 - min_distances^2)
    
    # Compute RBF features with shifted distances
    rbf_features = exp.(-distance_matrix_shifted / (2 * d^2))
    
    # Normalize each column (each time point)
    rbf_features_normalized = rbf_features ./ sum(rbf_features, dims=1)
    
    # Construct operator: K = Φ(t+1) * pinv(Φ(t))
    # where Φ is the normalized RBF features
    # For numerical stability with wide matrices, use: K = Φ(t+1) * Φ(t)' * pinv(Φ(t) * Φ(t)')
    if N > 1
        Φ_t = rbf_features_normalized[:, 1:end-1]  # Shape: (3, N-1)
        Φ_tp1 = rbf_features_normalized[:, 2:end]   # Shape: (3, N-1)
        
        # More numerically stable: compute K = Φ(t+1) * Φ(t)' * pinv(Φ(t) * Φ(t)')
        # This gives a 3x3 matrix instead of trying to invert a wide matrix
        G = Φ_t * Φ_t'  # Shape: (3, 3) - Gram matrix
        A = Φ_tp1 * Φ_t'  # Shape: (3, 3) - cross-correlation matrix
        
        # Use pinv with tolerance to handle potential rank deficiency
        rbf_operator = nothing
        try
            rbf_operator = A * pinv(G; rtol=1e-10)
        catch e
            @warn "pinv failed for alpha=$alpha, trying with higher tolerance: $e"
            try
                rbf_operator = A * pinv(G; rtol=1e-6)
            catch e2
                @error "pinv failed even with relaxed tolerance for alpha=$alpha: $e2"
                @warn "Skipping alpha=$alpha"
                continue  # Skip this alpha value
            end
        end
        
        if rbf_operator === nothing
            @warn "Skipping alpha=$alpha due to computation failure"
            continue
        end
        
        # Compute eigenvalues and eigenvectors
        Λ, V = eigen(rbf_operator)
        
        rbf_operators[alpha] = rbf_operator
        rbf_generators[alpha] = (rbf_operator - I)/ dt
        rbf_eigenvalues[alpha] = Λ
        rbf_eigenvectors[alpha] = V
        
        @info "RBF operator for alpha = $alpha: size $(size(rbf_operator))"
    end
end

@info "Saving results"
output_file = joinpath(data_directory, "dictionary_operators.hdf5")
hfile = h5open(output_file, "w")

# Save Ulam's method results
hfile["ulam_markov_chain"] = ulam_markov_chain
hfile["ulam_onehot"] = ulam_onehot  # One-hot encoding: (N, 3) matrix
hfile["ulam_generator"] = Q_ulam
hfile["ulam_perron_frobenius"] = P_ulam  # Save P for reference
hfile["ulam_unique_states"] = 3  # left lobe, origin, right lobe

# Save distance information
hfile["distances_origin"] = distances_origin
hfile["distances_left"] = distances_left
hfile["distances_right"] = distances_right
hfile["fixed_point_origin"] = fp_origin
hfile["fixed_point_left"] = fp_left
hfile["fixed_point_right"] = fp_right

# Save RBF results
for alpha in alphas
    if haskey(rbf_operators, alpha)
        hfile["rbf_operator_alpha_$alpha"] = rbf_operators[alpha]
        hfile["rbf_generator_alpha_$alpha"] = rbf_generators[alpha]
        hfile["rbf_eigenvalues_alpha_$alpha"] = rbf_eigenvalues[alpha]
        hfile["rbf_eigenvectors_alpha_$alpha"] = rbf_eigenvectors[alpha]
    end
end

hfile["dt"] = dt
hfile["r"] = r
hfile["sigma"] = sigma
hfile["beta"] = beta

close(hfile)

@info "Done! Results saved to $output_file"
@info "Ulam's method: 3 unique states (left lobe=1, origin=2, right lobe=3)"
for alpha in alphas
    if haskey(rbf_operators, alpha)
        @info "RBF alpha=$alpha: operator size $(size(rbf_operators[alpha]))"
    end
end


for alpha in alphas
    display(rbf_operators[alpha])
end
display(P_ulam)

for alpha in alphas
    d = 10^(-alpha * 1.0)
    println("d = $d")
    display(rbf_generators[alpha])
end
display(Q_ulam)

for alpha in alphas
    d = 10^(-alpha * 1.0)
    println("operator norm for d = $d")
    display(norm(rbf_generators[alpha] - Q_ulam))
end
