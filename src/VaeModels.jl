module VaeModels
using Flux, MLDatasets
using Flux: DataLoader, params, pullback, logitbinarycrossentropy
using BSON: @load, @save
using TensorBoardLogger, Logging, ProgressLogging
using LinearAlgebra: diagm

include("base.jl")
include("architectures.jl")
include("losses.jl")
include("trainingloops.jl")

export VaeEncoder, FullVae
export makeMNISTVae
export VAEloss_unitarycoherence
export trainVae, trainstdVaeonMNIST

end # module VaeModels
