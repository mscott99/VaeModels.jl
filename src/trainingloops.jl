# TODO remove loss from training loop.
function trainVae(β, λ, model, pars::Flux.Params, traindata, opt::Flux.Optimise.AbstractOptimiser, numepochs, savedir, tblogdir; loginterval=100, saveinterval=1, label="")
    # The training loop for the model
    tblogger = TBLogger(tblogdir)
    saveindex = 0

    function savemodelcheckpoint()
        @save string(savedir, label, "intrain", saveindex) model opt
        saveindex += 1
    end

    function klfromgaussian(μ, logvar)
        0.5 * sum(@. (exp(logvar) + μ^2 - logvar - 1.0))
    end

    function l2reg(pars) #probably make this again with @functor
        sum(x -> sum(abs2, x), pars)
    end

    #numbatches = length(data)
    @progress for epochnum in 1:numepochs
        for (step, x) in enumerate(traindata)
            with_logger(tblogger) do
                loss, back = pullback(pars) do
                    intermediate = model.encoder.encoderbody(x)
                    μ = model.encoder.splitedμ(intermediate)
                    logvar = model.encoder.splitedlogvar(intermediate)
                    z = μ .+ randn(Float32, size(logvar)) .* exp.(0.5f0 .* logvar)
                    x̂ = model.decoder(z)
                    mainloss = logitbinarycrossentropy(x̂, x; agg=sum)
                    KL = β * klfromgaussian(μ, logvar)
                    Reg = λ * l2reg(pars)
                    @info "losspieces" mainloss KL Reg
                    mainloss + KL + Reg
                end
                gradients = back(1.0f0)
                Flux.Optimise.update!(opt, pars, gradients)

                if step % loginterval == 0
                    with_logger(tblogger) do
                        @info "loss" loss
                    end
                end
            end

            # if step % saveinterval == 0
            #     savemodel()
            #     @save string(savedir, label, "_epoch_", epochnum) model opt
            # end

        end
        if epochnum % saveinterval == 0
            savemodelcheckpoint()
        end
    end
    savemodelcheckpoint()
    @info "training complete!"
end

function modularizedtrainVae(β, λ, model, pars::Flux.Params, η, traindata, numepochs, savedir, tblogdir; loginterval=100, saveinterval=1, label="")
    # The training loop for the model
    tblogger = TBLogger(tblogdir)
    saveindex = 0

    function savemodelcheckpoint()
        @save string(savedir, label, "intrain", saveindex) model opt
        saveindex += 1
    end

    opt = Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0f0), Optimisers.Adam(η))
    opt_state = Optimisers.setup(opt, model)

    #numbatches = length(data)
    @progress for epochnum in 1:numepochs
        for (step, x) in enumerate(traindata)

            loss, gradients = Flux.withgradient(Vaeloss, model, x, β, λ, pars)
            if !isfinite(loss)
                @warn "Found invalid loss, skipping step."
                continue # here I would like to clip the gradients
            end
            opt_state, model = Optimisers.update!(opt_state, model, gradients[1])

            if step % loginterval == 0
                with_logger(tblogger) do
                    @info "loss" loss
                end
            end


            # if step % saveinterval == 0
            #     savemodel()
            #     @save string(savedir, label, "_epoch_", epochnum) model opt
            # end
        end
        if epochnum % saveinterval == 0
            savemodelcheckpoint()
        end
    end
    savemodelcheckpoint()
    @info "training complete!"
end

function trainstdVaeonMNIST(; numepochs=10, pretrained_dir="./pretrained", logdir="./logs/", kwargs...)
    model = makeMNISTVae(512, 512, 16)
    batchsize = 32 #debugging
    traindata = MNIST(Float32, :train).features # debugging
    trainloader = DataLoader(traindata, batchsize=batchsize)

    trainVae(1.0f0, 0.01f0, model, params(model), trainloader, Adam(), numepochs, pretrained_dir, logdir, label="working", loginterval=100; kwargs...)
end