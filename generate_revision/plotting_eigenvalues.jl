using CairoMakie

hfile = h5open(data_directory  * "/structured_eigenvalues_and_generators.hdf5", "r")
Λ_bskmeans = read(hfile["Λ_bskmeans"])
Λ_unstructured = read(hfile["Λ_unstructured"])
Λ_kmeans = read(hfile["Λ_kmeans"])
Λ_rbf = read(hfile["Λ_rbf"])
Λ_rbf_2 = read(hfile["Λ_rbf_2"])
Λ_rbf_3 = read(hfile["Λ_rbf_3"])
Λ_ulam = read(hfile["Λ_ulam"])
close(hfile)

fig = Figure(resolution = (600, 600))
xs = (-0.1, 1.1)# (-1.1, 1.1)
ys = (-0.6, 0.6)# (-1.1, 1.1)
ax = Axis(fig[1, 1]; title = "K-Means", ylabel = "Imaginary", xlabel = "Real")
bkmeans_indices = reverse(sortperm(abs.(Λ_bskmeans)))[1:256]
bkm_op = 0.25
scatter!(ax, real.(Λ_bskmeans[bkmeans_indices]), imag.(Λ_bskmeans[bkmeans_indices]), color = (:red, bkm_op))
scatter!(ax, real.(Λ_kmeans), imag.(Λ_kmeans), color = (:blue, 0.5))
xlims!(ax, xs)
ylims!(ax, ys)
hidexdecorations!(ax; grid = false)
ax = Axis(fig[1, 2]; title = "RBF", ylabel = "Imaginary", xlabel = "Real")
scatter!(ax, real.(Λ_bskmeans[bkmeans_indices]), imag.(Λ_bskmeans[bkmeans_indices]), color = (:red, bkm_op))
scatter!(ax, real.(Λ_rbf_3), imag.(Λ_rbf_3), color = (:blue, 0.5))
xlims!(ax, xs)
ylims!(ax, ys)
hidexdecorations!(ax; grid = false)
hideydecorations!(ax; grid = false)
ax = Axis(fig[2, 1]; title = "Bisecting K-Means", ylabel = "Imaginary", xlabel = "Real")
scatter!(ax, real.(Λ_bskmeans[bkmeans_indices]), imag.(Λ_bskmeans[bkmeans_indices]), color = (:red, bkm_op))
scatter!(ax, real.(Λ_unstructured), imag.(Λ_unstructured), color = (:blue, 0.5))
xlims!(ax, xs)
ylims!(ax, ys)
ax = Axis(fig[2, 2]; title = "Ulam", ylabel = "Imaginary", xlabel = "Real")
scatter!(ax, real.(Λ_bskmeans[bkmeans_indices]), imag.(Λ_bskmeans[bkmeans_indices]), color = (:red, bkm_op))
scatter!(ax, real.(Λ_ulam), imag.(Λ_ulam), color = (:blue, 0.5))
xlims!(ax, xs)
ylims!(ax, ys)
hideydecorations!(ax; grid = false)

save("RevisionFigures/eigenvalues.png", fig)

##
hfile = h5open(data_directory  * "/structured_eigenvalues_and_generators.hdf5", "r")
s_kmeans = read(hfile["s_kmeans"])
s_rbf_3 = read(hfile["s_rbf_3"])
s_unstructured = read(hfile["s_unstructured"])
s_ulam = read(hfile["s_ulam"])
pratio_kmeans = read(hfile["pratio_kmeans"])
pratio_rbf_3 = read(hfile["pratio_rbf_3"])
pratio_ulam = read(hfile["pratio_ulam"])
pratio_unstructured = read(hfile["pratio_unstructured"])
rbf_operator_time = read(hfile["rbf_operator_time_2"])
pf_bskmeans_time = read(hfile["pf_bskmeans_time"])
pf_kmeans_time = read(hfile["pf_kmeans_time"])
close(hfile)

# Print out second largest purely real eigenvalue for each method
ind_real = argmax(reverse(imag.(Λ_kmeans)[1:end-1]) .== 0) 
lambda_kmeans = Λ_kmeans[end-ind_real]
ind_real = argmax(reverse(imag.(Λ_rbf_3)[1:end-1]) .== 0) 
lambda_rbf_3 = Λ_rbf_3[end-ind_real]
ind_real = argmax(reverse(imag.(Λ_unstructured)[1:end-1]) .== 0) 
lambda_unstructured = Λ_unstructured[end-ind_real]
ind_real = argmax(reverse(imag.(Λ_ulam)[1:end-1]) .== 0) 
lambda_ulam = Λ_ulam[end-ind_real]
ind_real = argmax(reverse(imag.(Λ_bskmeans)[1:end-1]) .== 0) 
lambda_bskmeans = Λ_bskmeans[end-ind_real]

print(lambda_kmeans)
print(lambda_rbf_3)
print(lambda_unstructured)
print(lambda_ulam)
print(lambda_bskmeans)

hfile = h5open(data_directory  * "/structured_embedding_revision.hdf5", "r")
ulam_time = read(hfile["ulam_time"])
kmeans_time = read(hfile["kmeans_time"])
unstructured_time = read(hfile["unstructured_time"])
close(hfile)

print(ulam_time)
print(kmeans_time)
print(unstructured_time)
print(kmeans_time + 5.031185865402222 ) # RBF time