struct VaeEncoder{T,V,L}
    encoderbody::T
    splitedμ::V
    splitedlogvar::L
end
Flux.@functor VaeEncoder

function (m::VaeEncoder)(x::AbstractArray)
    intermediate = m.encoderbody(x)
    μ = m.splitedμ(intermediate)
    logvar = m.splitedlogvar(intermediate)
    return μ, logvar
end

struct FullVae{T}
    encoder::VaeEncoder
    decoder::T
end
Flux.@functor FullVae


#forward pass
function (m::FullVae)(x::Vector; rng=TaskLocalRNG())
    μ, logvar = m.encoder(x)
    randcoeffs = randn(rng, Float32, size(logvar))
    z = μ .+ randcoeffs .* exp.(0.5f0 .* logvar)
    m.decoder(z)
end


#averaged forward pass
function (m::FullVae)(x::Vector, n::Integer; rng=TaskLocalRNG())
    #preformance gain available by getting mu and logvar once, and sampling many times
    acc = zero(m(x))
    μ, logvar = m.encoder(x)
    #acc = zero(m.decoder(zeros(Float32, size(logvar))))

    for i in 1:n
        randcoeffs = randn(rng, Float32, size(logvar))
        z = μ .+ randcoeffs .* exp.(0.5f0 .* logvar)
        acc .+= m.decoder(z)
    end
    acc ./ n
end

function (m::FullVae)(x::Matrix; kwargs...)
    m(reshape(x, :); kwargs...)
end

function (m::FullVae)(x::Matrix, n::Integer; rng=TaskLocalRNG())
    #preformance gain available by getting mu and logvar once, and sampling many times
    m(reshape(x, :), n; rng=rng)
end
