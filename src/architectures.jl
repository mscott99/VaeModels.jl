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
