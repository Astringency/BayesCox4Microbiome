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

    if prior isa HorseshoePrior
        tau_scale = _resolved_tau_scale(prior, p, n_events)
        tau ~ truncated(Cauchy(0, tau_scale), 0.0, Inf)
        lambda ~ Turing.filldist(truncated(Cauchy(0, prior.local_scale), 0.0, Inf), p)
        beta ~ Turing.product_distribution(Normal.(0, tau .* lambda))
        if use_car && car_weights !== nothing
            Turing.@addlogprob! _car_logprior(log.(lambda), car_weights, car_precision)
        end
    elseif prior isa RegularizedHorseshoePrior
        if prior.sample_pseudo_sigma
            pseudo_sigma2 ~ _positive_t_distribution(
                prior.pseudo_sigma_df,
                prior.pseudo_sigma_scale,
            )
            pseudo_sigma = sqrt(pseudo_sigma2)
        else
            pseudo_sigma = prior.pseudo_sigma
        end
        tau_scale = _resolved_tau_scale(prior, data; pseudo_sigma = pseudo_sigma)
        tau_raw ~ truncated(Cauchy(0, 1.0), 0.0, Inf)
        tau = tau_scale * tau_raw
        c2 ~ InverseGamma(prior.slab_df / 2, prior.slab_df * prior.slab_s2 / 2)
        lambda ~ Turing.filldist(truncated(Cauchy(0, prior.local_scale), 0.0, Inf), p)
        lambda2 = lambda .^ 2
        lambda_tilde2 = @. c2 * lambda2 / (c2 + tau^2 * lambda2)
        beta ~ Turing.product_distribution(Normal.(0, tau .* sqrt.(lambda_tilde2)))
        if use_car && car_weights !== nothing
            Turing.@addlogprob! _car_logprior(log.(lambda), car_weights, car_precision)
        end
    elseif prior isa BayesianLassoPrior
        if prior.lambda === nothing
            lambda2 ~ Gamma(prior.lambda2_shape, 1 / prior.lambda2_rate)
        else
            lambda2 = prior.lambda^2
        end
        tau2 ~ Turing.filldist(Gamma(1.0, 2 / lambda2), p)
        beta ~ Turing.product_distribution(Normal.(0, sqrt.(tau2)))
    elseif prior isa SpikeSlabLassoPrior
        theta ~ Beta(prior.theta_a, prior.theta_b)
        beta ~ Turing.filldist(Laplace(0, prior.slab_scale), p)
        for j in 1:p
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

function _combine_callbacks(timing_callback, user_callback)
    user_callback === nothing && return timing_callback
    return (args...; kwargs...) -> begin
        timing_callback(args...; kwargs...)
        user_callback(args...; kwargs...)
        return nothing
    end
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
    progress::Bool = true,
    callback = nothing,
)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive."))
    warmup >= 0 || throw(ArgumentError("warmup must be nonnegative."))
    chains > 0 || throw(ArgumentError("chains must be positive."))
    _validate_prior(prior)

    p = size(data.z, 2)
    scale = soft_center_scale === nothing ? _default_soft_center_scale(p) : Float64(soft_center_scale)
    resolved_car_weights = use_car ? _prepare_car_weights(car_weights, p) : nothing
    model = _bayes_cox_turing_model(
        data,
        prior,
        soft_center,
        scale,
        resolved_car_weights,
        use_car && resolved_car_weights !== nothing,
        Float64(car_precision),
    )
    sampler = Turing.NUTS(warmup, Float64(target_accept))
    start_ns = time_ns()
    first_saved_times = Float64[]
    timing_lock = ReentrantLock()
    timing_callback = (rng_cb, model_cb, sampler_cb, sample_cb, state_cb, iteration; kwargs...) -> begin
        if iteration == 1
            lock(timing_lock)
            try
                push!(first_saved_times, (time_ns() - start_ns) / 1e9)
            finally
                unlock(timing_lock)
            end
        end
        return nothing
    end
    combined_callback = _combine_callbacks(timing_callback, callback)

    timed = @timed if chains == 1
        Turing.sample(rng, model, sampler, n_samples; progress = progress, callback = combined_callback)
    else
        Turing.sample(
            rng,
            model,
            sampler,
            Turing.MCMCThreads(),
            n_samples,
            chains;
            progress = progress,
            callback = combined_callback,
        )
    end
    warmup_seconds =
        warmup == 0 ? 0.0 :
        isempty(first_saved_times) ? missing : min(maximum(first_saved_times), timed.time)
    sample_seconds =
        warmup_seconds === missing ? missing : max(timed.time - warmup_seconds, 0.0)
    gc_percent = timed.time > 0 ? 100 * timed.gctime / timed.time : NaN
    diagnostics = SamplingDiagnostics(
        warmup_seconds,
        sample_seconds,
        timed.time,
        Int64(timed.bytes),
        timed.gctime,
        gc_percent,
    )
    return (samples = timed.value, diagnostics = diagnostics)
end
