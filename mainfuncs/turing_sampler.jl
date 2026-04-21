const _TURING_MODEL_DEFINED = Ref(false)

function _require_turing()
    try
        @eval import Turing
    catch err
        throw(ArgumentError(
            "sampler=:nuts requires Turing.jl. Install it with `import Pkg; Pkg.add(\"Turing\")`, " *
            "or call `sample_bayes_cox(...; sampler=:mh)` to use the built-in fallback sampler. " *
            "Original error: $(typeof(err)): $err",
        ))
    end
    return nothing
end

function _ensure_turing_model!()
    _TURING_MODEL_DEFINED[] && return nothing
    _require_turing()

    @eval begin
        Turing.@model function _bayes_cox_turing_model(
            data::CoxPreparedData,
            prior::AbstractCoxPrior,
            soft_center::Bool,
            soft_center_scale::Float64,
            car_weights,
            use_car::Bool,
            car_precision::Float64,
        )
            p = size(data.z, 2)
            n_events = count(!=(0), data.event)
            beta = Vector{Real}(undef, p)

            if prior isa HorseshoePrior
                tau_scale = _resolved_tau_scale(prior, p, n_events)
                tau ~ truncated(Cauchy(0, tau_scale), 0.0, Inf)
                lambda = Vector{Real}(undef, p)
                for j in 1:p
                    lambda[j] ~ truncated(Cauchy(0, prior.local_scale), 0.0, Inf)
                    beta[j] ~ Normal(0, tau * lambda[j])
                end
                if use_car && car_weights !== nothing
                    Turing.@addlogprob! _car_logprior(log.(lambda), car_weights, car_precision)
                end
            elseif prior isa RegularizedHorseshoePrior
                if prior.sample_pseudo_sigma
                    pseudo_sigma ~ _positive_t_distribution(
                        prior.pseudo_sigma_df,
                        prior.pseudo_sigma_scale,
                    )
                else
                    pseudo_sigma = prior.pseudo_sigma
                end
                tau_scale = _resolved_tau_scale(prior, data; pseudo_sigma = pseudo_sigma)
                tau_raw ~ truncated(Cauchy(0, 1.0), 0.0, Inf)
                tau = tau_scale * tau_raw
                c2 ~ InverseGamma(prior.slab_df / 2, prior.slab_df * prior.slab_s2 / 2)
                lambda = Vector{Real}(undef, p)
                for j in 1:p
                    lambda[j] ~ truncated(Cauchy(0, prior.local_scale), 0.0, Inf)
                    lambda_tilde2 = c2 * lambda[j]^2 / (c2 + tau^2 * lambda[j]^2)
                    beta[j] ~ Normal(0, tau * sqrt(lambda_tilde2))
                end
                if use_car && car_weights !== nothing
                    Turing.@addlogprob! _car_logprior(log.(lambda), car_weights, car_precision)
                end
            elseif prior isa BayesianLassoPrior
                if prior.lambda === nothing
                    lambda2 ~ Gamma(prior.lambda2_shape, 1 / prior.lambda2_rate)
                else
                    lambda2 = prior.lambda^2
                end
                tau2 = Vector{Real}(undef, p)
                for j in 1:p
                    tau2[j] ~ Gamma(1.0, 2 / lambda2)
                    beta[j] ~ Normal(0, sqrt(tau2[j]))
                end
            elseif prior isa SpikeSlabLassoPrior
                theta ~ Beta(prior.theta_a, prior.theta_b)
                for j in 1:p
                    beta[j] ~ Laplace(0, prior.slab_scale)
                    Turing.@addlogprob!(
                        _ssl_marginal_logpdf(beta[j], theta, prior.spike_scale, prior.slab_scale) -
                        _laplace_logpdf(beta[j], prior.slab_scale)
                    )
                end
            else
                error("unsupported prior type: $(typeof(prior))")
            end

            if soft_center
                Turing.@addlogprob! _soft_center_logpdf(beta, soft_center_scale)
            end
            Turing.@addlogprob! cox_logpartiallikelihood(data, beta)
            return beta
        end
    end

    _TURING_MODEL_DEFINED[] = true
    return nothing
end

function _sample_nuts(
    data::CoxPreparedData,
    prior::AbstractCoxPrior;
    n_samples::Integer = 5000,
    warmup::Integer = 2000,
    chains::Integer = 4,
    target_accept::Real = 0.8,
    rng::AbstractRNG = Random.default_rng(),
    soft_center::Bool = true,
    soft_center_scale::Union{Nothing,Real} = nothing,
    car_weights = nothing,
    use_car::Bool = car_weights !== nothing,
    car_precision::Real = 1.0,
)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive."))
    warmup >= 0 || throw(ArgumentError("warmup must be nonnegative."))
    chains > 0 || throw(ArgumentError("chains must be positive."))
    _validate_prior(prior)
    _ensure_turing_model!()

    p = size(data.z, 2)
    scale = soft_center_scale === nothing ? _default_soft_center_scale(p) : Float64(soft_center_scale)
    model = Base.invokelatest(
        _bayes_cox_turing_model,
        data,
        prior,
        soft_center,
        scale,
        car_weights,
        use_car,
        Float64(car_precision),
    )
    sampler = Base.invokelatest(Turing.NUTS, warmup, Float64(target_accept))

    if chains == 1
        return Base.invokelatest(Turing.sample, rng, model, sampler, n_samples)
    else
        serial = Base.invokelatest(Turing.MCMCSerial)
        return Base.invokelatest(Turing.sample, rng, model, sampler, serial, n_samples, chains)
    end
end
