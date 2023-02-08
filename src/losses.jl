function klfromgaussian(μ, logvar)
    0.5 * sum(@. (exp(logvar) + μ^2 - logvar - 1.0))
end

function l2reg(pars) #probably make this again with @functor
    sum(x -> sum(abs2, x), pars)
end

function Vaeloss(model, x, β, λ, pars)
    intermediate = model.encoder.encoderbody(x)
    μ = model.encoder.splitedμ(intermediate)
    logvar = model.encoder.splitedlogvar(intermediate)
    z = μ .+ randn(Float32, size(logvar)) .* exp.(0.5f0 .* logvar)
    x̂ = model.decoder(z)
    logitbinarycrossentropy(x̂, x; agg=sum) + β * klfromgaussian(μ, logvar) + λ * l2reg(pars)
end

loss_α(F, A) = maximum(sqrt.(sum((F * A) .* (F * A), dims=2))) + 100 * norm(A' * A - I(784), 2)^2

optimal_loss_α(F, A) = sum(abs2, (sqrt.(sum((F * A) .^ 2, dims=2)))) + 100 * norm(A' * A - I(784), 2)^2


# I do not know how

