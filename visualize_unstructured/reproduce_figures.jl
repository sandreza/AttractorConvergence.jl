data_directory = "/test_data"

figure_directory = pwd() * "/unstructured_figures"
isdir(figure_directory) ? nothing : mkdir(figure_directory)

@info "creating figures"

figure_number = 1
include("state_space_partitions.jl")
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)
println("done with ", figure_number)
figure_number += 1

include("quasi_invariant_set.jl")
save(figure_directory * "/Figure" * string(figure_number) * ".png", fig)
println("done with ", figure_number)
figure_number += 1