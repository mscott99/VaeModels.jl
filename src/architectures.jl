function makeMNISTVae(hidden, secondhidden, zlayer)
    FullVae(
        VaeEncoder(
            Chain(
                Flux.flatten,
                Dense(28^2 => hidden, relu),
                Dense(hidden => secondhidden)
            ),
            Dense(secondhidden => zlayer),
            Dense(secondhidden => zlayer)
        ),
        Chain(
            Dense(zlayer => secondhidden, bias=false, relu),
            Dense(secondhidden => hidden, bias=false, relu),
            Dense(hidden => 28^2, bias=false),
            x -> reshape(x, 28, 28, :)
        )
    )
end

function makeconvMNISTVae()
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

    deepconvVae = FullVae(encoder, decoder)
end

function makebetterconvMNISTVae()
    # input is 28x28x1
    encoder = VaeEncoder(
        Chain(
            Conv((3, 3), 1 => 16, relu, pad=1), # 28x28 with number of weights 16x3x3x1 = 144
            MaxPool((2, 2)),  # 14x14 with number of weights 16x2x2 = 64
            Conv((3, 3), 16 => 16, relu, pad=1), # 14x14 with number of weights 32x3x3x16 = 4608
            MaxPool((2, 2)), # output size 7x7
            x -> Flux.flatten(x), # 1568
            Dense(784 => 256, relu)
        ),
        Dense(256 => 16, identity), # μ
        Dense(256 => 16, relu) # logvar
    )
    #Conv((3, 3), 8 => 4, relu, stride=7, pad=SamePad()),
    #Conv((3, 3), 8 => 4, relu, stride=7, pad=SamePad()),
    decoder = Chain(
        Dense(16 => 256, relu),
        Dense(256 => 7 * 7 * 16, relu),
        x -> reshape(x, 7, 7, 16, :),
        ConvTranspose((3, 3), 16 => 16, relu, pad=1), # 7x7
        Upsample((2, 2)), # 14x14
        ConvTranspose((3, 3), 16 => 16, relu, pad=1), # 14x14
        Upsample((2, 2)), # 28x28
        ConvTranspose((3, 3), 16 => 1, identity, pad=1), # 28x28
    )

    #decoder = Chain(
    #
    #        ConvTranspose((2, 2), 4 => 8, relu, stride=7, pad=SamePad()),
    #        ConvTranspose((7, 7), 8 => 8, relu, stride=2, pad=SamePad()),
    #        ConvTranspose((7, 7), 8 => 1, stride=2, pad=SamePad())
    #   )

    FullVae(encoder, decoder)
end
