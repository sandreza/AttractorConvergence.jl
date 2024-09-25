using MarkovChainHammer, ProgressBars, LinearAlgebra
using CairoMakie, HDF5

data_directory = "/nobackup1/sandre/AttractorConvergence/old_data/"
data_directory = "/nobackup1/sandre/AttractorConvergence/data/"

figure_directory = pwd() * "/unstructured_figures"
isdir(figure_directory) ? nothing : mkdir(figure_directory)

@info "creating figures"

figure_number = 0
include("splitting.jl")
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)
println("done with ", figure_number)
figure_number += 1

include("state_space_partitions.jl")
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)
println("done with ", figure_number)
figure_number += 1

include("entropy.jl")
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)
println("done with ", figure_number)
figure_number += 1

include("steady_state_xy.jl")
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)
println("done with ", figure_number)
figure_number += 1


include("steady_state.jl")
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)
println("done with ", figure_number)
figure_number += 1

include("autocorrelations.jl")
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)
println("done with ", figure_number)
figure_number += 1

include("quasi_invariant_set.jl")
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)
println("done with ", figure_number)
figure_number += 1

include("koopman_timeseries.jl")
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)
println("done with ", figure_number)
figure_number += 1
