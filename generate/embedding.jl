# open Lorenz data 
hfile = h5open(pwd() * data_directory  * "/lorenz.hdf5", "r")
timeseries = read(hfile["timeseries"])
s_timeseries = read(hfile["symmetrized timeseries"])
Δt = read(hfile["dt"])
close(hfile)
# read centers list from file 
hfile = h5open(pwd() * data_directory  * "/kmeans.hdf5", "r")
centers_matrix= read(hfile["centers"])
levels = read(hfile["levels"])
close(hfile)
centers_list = [[centers_matrix[:,1, i], centers_matrix[:,2, i]] for i in 1:size(centers_matrix)[3]]
# constructing embedding with 2^levels number of states
# note that we can also choose a number less than levels
embedding = StateTreeEmbedding(centers_list, levels)
##
@info "computing markov embedding"
markov_chain = zeros(Int64, size(timeseries)[2])
for i in ProgressBar(1:size(timeseries)[2])
    state = timeseries[:, i]
    # partition state space according to most similar markov state
    # This will be sped up by using a tree structure
    markov_i = embedding(state)
    @inbounds markov_chain[i] = markov_i
end
@info "computing symmetric embedding"
s_markov_chain = zeros(Int64, size(s_timeseries)[2])
for i in ProgressBar(1:size(s_timeseries)[2])
    state = s_timeseries[:, i]
    # partition state space according to most similar markov state
    # This will be sped up by using a tree structure
    markov_i = embedding(state)
    @inbounds s_markov_chain[i] = markov_i
end
##
@info "saving"
hfile = h5open(pwd() * data_directory  * "/embedding.hdf5", "w")
hfile["markov_chain"] = markov_chain
hfile["symmetrized markov chain"] = s_markov_chain
hfile["dt"] = Δt
close(hfile)