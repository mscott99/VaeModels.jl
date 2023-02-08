# using VaeModels

# trainstdVaeonMNIST(numepochs=1, pretrained_dir="../pretrained", logdir="../logs/")
#using GenerativeRecoveries
#using GenerativeRecoveries: setupCELEBAimagesignals, showCELEBAimage, setupMNISTimagesignals, getCELEBAdataset
using Images: colorview, RGB, Gray
using VaeModels: FullVae, VaeEncoder
using VaeModels
using Revise
using Flux: ConvTranspose, Conv, SamePad, Chain, flatten
using MLDatasets: MNIST
using Flux

#images = setupCELEBAimagesignals(2)
#images = setupMNISTimagesignals([1, 2])
#celdataset = getCELEBAdataset()

images = MNIST().features[:, :, 1:5];
images = reshape(images, 28, 28, 1, :);

#layer = ConvTranspose((5, 5), 1 => 3, stride=2, pad=SamePad())

#convlayer = Conv((5, 5), 1 => 3, stride=2, pad=SamePad())

#= 
encoder = VaeEncoder(
    Chain(
        Conv((7, 7), 1 => 8, relu, stride=2, pad=SamePad()),
        Conv((7, 7), 8 => 8, relu, stride=2, pad=SamePad())
    ),
    Conv((2, 2), 8 => 4, relu, stride=7, pad=SamePad()),
    Conv((2, 2), 8 => 4, relu, stride=7, pad=SamePad()),
)

decoder = Chain(
    ConvTranspose((2, 2), 4 => 8, relu, stride=7, pad=SamePad()),
    ConvTranspose((7, 7), 8 => 8, relu, stride=2, pad=SamePad()),
    ConvTranspose((7, 7), 8 => 1, stride=2, pad=SamePad())
)

deepconvVae = FullVae(encoder, decoder); =#
#deepconvVae(images, rng=TaskLocalRNG());



using VaeModels: modularizedtrainVae, Vaeloss
using Flux: params, DataLoader, Flux
using Random: TaskLocalRNG

#pars = params(deepconvVae);
#try these parameters next time
#modularizedtrainVae(0.001f0, 0.00001f0, deepconvVae, pars, 1.0f-3, trainloader, 2, "./pretrained/", "./logs/"; loginterval=100, label="testing3_")

using BSON: @load
using Flux
using VaeModels
using Random: TaskLocalRNG
using Images: Gray
using MLDatasets: MNIST

dataset = reshape(MNIST(:train).features, 28, 28, 1, :);
trainloader = DataLoader(dataset, batchsize=32);
@load "./pretrained/testing3_intrain40" model

using GenerativeRecoveries
using GenerativeRecoveries: plot_MNISTrecoveries

displayimage = first(trainloader)[:, :, 1, 6]
nnimage = first(trainloader)[:, :, :, 6:6]

plot_MNISTrecoveries(model, [16, 512], [2, 4])

Gray.(sigmoid(model(nnimage, 1000; rng=TaskLocalRNG()))[:, :, 1, 1])

Gray.(displayimage)

using GenerativeRecoveries: recoversignal, inversesigmoid

using LinearAlgebra: I

struct batchIdentity
end

import Base: *
*(b::batchIdentity, ar::AbstractArray) = I * ar[:, :, 1]
b = batchIdentity()
recovery = recoversignal(inversesigmoid.(displayimage), b, deepconvVae.decoder; init_code=randn((1, 1, 4, 1)))
Gray.(sigmoid(recovery[:, :, 1, 1]))
Gray.(sigmoid(inversesigmoid.(displayimage)))

using Flux: pullback
batch = first(trainloader)
pullback(() -> Vaeloss(batch, 1.0f0, 1.0f0, deepconvVae, pars), pars)

using Random: TaskLocalRNG
deepconvVae(batch, rng=TaskLocalRNG())

using BSON: @load
using VaeModels
using Flux
using Flux: DataLoader
using MLDatasets: MNIST
using Random: TaskLocalRNG
using Images: Gray

@load "./pretrained/firsttryintrain40" model

dataset = reshape(MNIST(:train).features, 28, 28, 1, :);
trainloader = DataLoader(dataset, batchsize=32);

displayimage = first(trainloader)[:, :, 1, 2]
nnimage = first(trainloader)[:, :, :, 2:2]
outimg = model(nnimage, 1000, rng=TaskLocalRNG())[:, :, 1, 1]
Gray.(outimg)
Gray.(displayimage)

using BSON: @load
@load "./pretrained/testconvintrain20" model