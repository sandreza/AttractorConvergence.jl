# open Lorenz data 
hfile = h5open(pwd() * "/data/lorenz.hdf5", "r")
timeseries = read(hfile["timeseries"])
s_timeseries = read(hfile["symmetrized timeseries"])
Δt = read(hfile["dt"])
close(hfile)
# ideally need to read centers_list from file, but right now it's not saved and the format isn't gonna work for saving 

# constructing embedding with 2^levels number of states
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
hfile = h5open(pwd() * "/data/embedding.hdf5", "w")
hfile["markov_chain"] = markov_chain
hfile["symmetrized markov chain"] = s_timeseries
hfile["dt"] = Δt
close(hfile)