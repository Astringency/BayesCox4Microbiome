Turing.@model function _bayes_cox_turing_model(
    data::CoxPreparedData,
    prior::AbstractCoxPrior,
    constraint::Symbol,
    prox_sum0_lambda::Float64,
    car_weights,
    use_car::Bool,
    car_precision::Float64,
    sample_car_strength::Bool,
    car_strength_prior_a::Float64,
    car_strength_prior_b::Float64,
    car_degree_scale_value::Float64,
    car_rank::Int,
    noncentered::Bool,
)
    p = size(data.z, 2)
    n_events = count(!=(0), data.event)

    if prior isa HorseshoePrior
        tau_cauchy_scale = _global_tau_cauchy_scale(_resolved_tau_scale(prior, p, n_events))
        tau ~ truncated(Cauchy(0, tau_cauchy_scale), 0.0, Inf)
        lambda ~ Turing.filldist(truncated(Cauchy(0, prior.local_scale), 0.0, Inf), p)
        if noncentered
            beta_raw ~ Turing.filldist(Normal(0, 1), p)
            beta = tau .* lambda .* beta_raw
        else
            beta ~ Turing.product_distribution(Normal.(0, tau .* lambda))
        end
        if use_car && car_weights !== nothing
            if sample_car_strength
                logit_car_kappa ~ Normal(0, 1)
                car_kappa = _logistic(logit_car_kappa)
                rho = car_kappa / car_degree_scale_value
                car_lp = _car_logprior(
                    log.(lambda),
                    car_weights,
                    rho,
                )
                if car_rank > 0
                    car_lp += 0.5 * car_rank * log(rho)
                end
                car_lp += _beta_logit_logpdf(
                    logit_car_kappa,
                    car_strength_prior_a,
                    car_strength_prior_b,
                ) - logpdf(Normal(0, 1), logit_car_kappa)
                Turing.@addlogprob! car_lp
            else
                Turing.@addlogprob! _car_logprior(
                    log.(lambda),
                    car_weights,
                    car_precision,
                )
            end
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
        tau_cauchy_scale = _global_tau_cauchy_scale(_resolved_tau_scale(prior, data; pseudo_sigma = pseudo_sigma))
        tau_raw ~ truncated(Cauchy(0, 1.0), 0.0, Inf)
        tau = tau_cauchy_scale * tau_raw
        c2 ~ InverseGamma(prior.slab_df / 2, prior.slab_df * prior.slab_s2 / 2)
        lambda ~ Turing.filldist(truncated(Cauchy(0, prior.local_scale), 0.0, Inf), p)
        if noncentered
            beta_raw ~ Turing.filldist(Normal(0, 1), p)
            beta = Vector{typeof(tau * beta_raw[1])}(undef, p)
            for j in 1:p
                log_beta_scale = _regularized_horseshoe_beta_logscale(tau, lambda[j], c2)
                beta[j] = exp(log_beta_scale) * beta_raw[j]
            end
        else
            beta ~ Turing.filldist(Normal(0, 1), p)
            for j in 1:p
                log_beta_scale = _regularized_horseshoe_beta_logscale(tau, lambda[j], c2)
                Turing.@addlogprob!(
                    _normal_zero_logpdf_from_logscale(beta[j], log_beta_scale) -
                    _std_normal_logpdf(beta[j])
                )
            end
        end
        if use_car && car_weights !== nothing
            if sample_car_strength
                logit_car_kappa ~ Normal(0, 1)
                car_kappa = _logistic(logit_car_kappa)
                rho = car_kappa / car_degree_scale_value
                car_lp = _car_logprior(
                    log.(lambda),
                    car_weights,
                    rho,
                )
                if car_rank > 0
                    car_lp += 0.5 * car_rank * log(rho)
                end
                car_lp += _beta_logit_logpdf(
                    logit_car_kappa,
                    car_strength_prior_a,
                    car_strength_prior_b,
                ) - logpdf(Normal(0, 1), logit_car_kappa)
                Turing.@addlogprob! car_lp
            else
                Turing.@addlogprob! _car_logprior(
                    log.(lambda),
                    car_weights,
                    car_precision,
                )
            end
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
        tau2_spike ~ Turing.filldist(Exponential(2 * prior.spike_scale^2), p)
        tau2_slab ~ Turing.filldist(Exponential(2 * prior.slab_scale^2), p)
        beta ~ Turing.filldist(Normal(0, 1), p)
        for j in 1:p
            Turing.@addlogprob!(
                _logsumexp2(
                    log1p(-theta) + _normal_zero_logpdf_from_logscale(beta[j], 0.5 * log(tau2_spike[j])),
                    log(theta) + _normal_zero_logpdf_from_logscale(beta[j], 0.5 * log(tau2_slab[j])),
                ) - _std_normal_logpdf(beta[j])
            )
        end
    else
        error("unsupported prior type: $(typeof(prior))")
    end

    Turing.@addlogprob! _constraint_logpenalty(
        beta;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    Turing.@addlogprob! cox_logpartiallikelihood(data, beta)
    return beta
end

function _chains_param_index(param_names::Vector{String}, name::AbstractString)
    idx = findfirst(==(String(name)), param_names)
    idx === nothing && throw(ArgumentError("NUTS output does not contain parameter $name."))
    return idx
end

function _noncentered_turing_samples_to_mhsamples(
    chains,
    data::CoxPreparedData,
    prior::Union{HorseshoePrior,RegularizedHorseshoePrior};
    constraint::Symbol,
    prox_sum0_lambda::Real,
    car_weights = nothing,
    use_car::Bool = car_weights !== nothing,
    car_precision::Real = 1.0,
    sample_car_strength::Bool = false,
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_degree_scale::Symbol = :mean_active,
    car_kappa_init::Real = 0.5,
)
    arr = _chains_value_array(chains)
    ndims(arr) == 2 && (arr = reshape(arr, size(arr, 1), size(arr, 2), 1))
    param_names = _chains_param_names(chains)
    p = size(data.z, 2)
    n_iter, _, n_chain = size(arr)
    output_names = _state_names(prior, p; sample_car_strength = sample_car_strength)
    raw_idx = [_chains_param_index(param_names, "beta_raw[$j]") for j in 1:p]
    lambda_idx = [_chains_param_index(param_names, "lambda[$j]") for j in 1:p]
    tau_idx = prior isa HorseshoePrior ? _chains_param_index(param_names, "tau") : nothing
    tau_raw_idx = prior isa RegularizedHorseshoePrior ? _chains_param_index(param_names, "tau_raw") : nothing
    c2_idx = prior isa RegularizedHorseshoePrior ? _chains_param_index(param_names, "c2") : nothing
    pseudo_idx = prior isa RegularizedHorseshoePrior && prior.sample_pseudo_sigma ?
        _chains_param_index(param_names, "pseudo_sigma2") : nothing
    logit_car_kappa_idx = sample_car_strength ?
        _chains_param_index(param_names, "logit_car_kappa") : nothing

    samples = Vector{MHSamples}(undef, n_chain)
    for c in 1:n_chain
        draws = Matrix{Float64}(undef, n_iter, length(output_names))
        lps = Vector{Float64}(undef, n_iter)
        for i in 1:n_iter
            u_nc = Vector{Float64}(undef, length(output_names))
            @inbounds for j in 1:p
                u_nc[j] = arr[i, raw_idx[j], c]
            end
            if prior isa HorseshoePrior
                u_nc[p + 1] = log(arr[i, tau_idx, c])
                @inbounds for j in 1:p
                    u_nc[p + 1 + j] = log(arr[i, lambda_idx[j], c])
                end
            else
                pseudo_sigma =
                    prior.sample_pseudo_sigma ? sqrt(arr[i, pseudo_idx, c]) : prior.pseudo_sigma
                tau_cauchy_scale = _global_tau_cauchy_scale(_resolved_tau_scale(prior, data; pseudo_sigma = pseudo_sigma))
                u_nc[p + 1] = log(tau_cauchy_scale * arr[i, tau_raw_idx, c])
                @inbounds for j in 1:p
                    u_nc[p + 1 + j] = log(arr[i, lambda_idx[j], c])
                end
                u_nc[2p + 2] = log(arr[i, c2_idx, c])
                if prior.sample_pseudo_sigma
                    u_nc[2p + 3] = log(arr[i, pseudo_idx, c])
                end
            end
            if sample_car_strength
                u_nc[end] = arr[i, logit_car_kappa_idx, c]
            end
            centered = _centered_state_from_noncentered(prior, data, u_nc)
            draws[i, :] .= centered
            lps[i] = logposterior(
                centered,
                data,
                prior;
                constraint = constraint,
                prox_sum0_lambda = prox_sum0_lambda,
                car_weights = car_weights,
                use_car = use_car,
                car_precision = car_precision,
                sample_car_strength = sample_car_strength,
                car_strength_prior_a = car_strength_prior_a,
                car_strength_prior_b = car_strength_prior_b,
                car_degree_scale = car_degree_scale,
                car_kappa_init = car_kappa_init,
            )
        end
        samples[c] = MHSamples(draws, lps, output_names, NaN, NaN, fill(NaN, length(output_names)))
    end
    return n_chain == 1 ? samples[1] : MultiMHSamples(samples)
end

Turing.@model function _lasso_prox_turing_model(
    data::CoxPreparedData,
    prior::BayesianLassoPrior,
    constraint::Symbol,
    prox_sum0_lambda::Float64,
    prox_l1_lambda::Float64,
    alpha_prior_shape::Float64,
    alpha_prior_rate::Float64,
    prox_root_tol::Float64,
    prox_root_maxiter::Int,
)
    p = size(data.z, 2)
    beta ~ Turing.filldist(Normal(0, 1), p)
    eta ~ Normal(0, 1)
    u = vcat(beta, [eta])
    lp = _lasso_prox_logposterior(
        u,
        data,
        prior;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_rate = alpha_prior_rate,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    Turing.@addlogprob! lp - sum(logpdf.(Normal(0, 1), beta)) - logpdf(Normal(0, 1), eta)
    return beta
end

Turing.@model function _ssl_prox_turing_block_model(
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior,
    gamma::AbstractVector{Bool},
    theta::Float64,
    constraint::Symbol,
    prox_sum0_lambda::Float64,
    prox_l1_lambda::Float64,
    alpha_prior_shape::Float64,
    alpha_prior_rate::Float64,
    prox_root_tol::Float64,
    prox_root_maxiter::Int,
)
    p = size(data.z, 2)
    beta ~ Turing.filldist(Normal(0, 1), p)
    eta ~ Normal(0, 1)
    u = vcat(beta, [eta])
    lp = _ssl_prox_logposterior(
        u,
        data,
        prior,
        gamma,
        theta;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_rate = alpha_prior_rate,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    Turing.@addlogprob! lp - sum(logpdf.(Normal(0, 1), beta)) - logpdf(Normal(0, 1), eta)
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
    constraint::Symbol = :soft_center,
    prox_sum0_lambda::Real = 1e-3,
    car_weights = nothing,
    use_car::Bool = car_weights !== nothing,
    car_precision::Real = 1.0,
    sample_car_strength::Bool = false,
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_degree_scale::Symbol = :mean_active,
    car_kappa_init::Real = 0.5,
    progress::Bool = true,
    callback = nothing,
    prox_l1_lambda::Real = 1e-3,
    alpha_prior_shape = nothing,
    alpha_prior_rate = nothing,
    prox_root_tol::Real = 1e-10,
    prox_root_maxiter::Integer = 100,
    thin::Integer = 1,
    noncentered::Bool = false,
)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive."))
    warmup >= 0 || throw(ArgumentError("warmup must be nonnegative."))
    chains > 0 || throw(ArgumentError("chains must be positive."))
    _validate_prior(prior)

    p = size(data.z, 2)
    resolved_constraint = _resolve_constraint(constraint)
    resolved_car_weights = use_car ? _prepare_car_weights(car_weights, p) : nothing
    if sample_car_strength
        _validate_adaptive_car_options(
            resolved_car_weights,
            car_strength_prior_a,
            car_strength_prior_b,
            car_kappa_init,
        )
    end
    car_degree_scale_value = sample_car_strength ?
        _car_degree_scale(resolved_car_weights; method = car_degree_scale) :
        1.0
    car_rank = sample_car_strength ? _car_rank(resolved_car_weights) : 0
    use_proximal_prior = _is_proximal_l1_prior(prior)
    if noncentered
        _check_noncentered_supported(prior)
    end
    if use_proximal_prior && prior isa BayesianLassoPrior
        return _sample_lasso_prox_nuts(
            data,
            prior;
            n_samples = n_samples,
            warmup = warmup,
            chains = chains,
            target_accept = target_accept,
            rng = rng,
            constraint = resolved_constraint,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = alpha_prior_shape,
            alpha_prior_rate = alpha_prior_rate,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
            progress = progress,
            callback = callback,
        )
    elseif use_proximal_prior && prior isa SpikeSlabLassoPrior
        return _sample_ssl_prox_nuts_gibbs(
            data,
            prior;
            n_samples = n_samples,
            warmup = warmup,
            thin = thin,
            chains = chains,
            target_accept = target_accept,
            rng = rng,
            constraint = resolved_constraint,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = alpha_prior_shape,
            alpha_prior_rate = alpha_prior_rate,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
            progress = progress,
        )
    end
    model = _bayes_cox_turing_model(
        data,
        prior,
        resolved_constraint,
        Float64(prox_sum0_lambda),
        resolved_car_weights,
        use_car && resolved_car_weights !== nothing,
        Float64(car_precision),
        sample_car_strength,
        Float64(car_strength_prior_a),
        Float64(car_strength_prior_b),
        Float64(car_degree_scale_value),
        Int(car_rank),
        noncentered,
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
    samples = noncentered ?
        _noncentered_turing_samples_to_mhsamples(
            timed.value,
            data,
            prior;
            constraint = resolved_constraint,
            prox_sum0_lambda = prox_sum0_lambda,
            car_weights = resolved_car_weights,
            use_car = use_car && resolved_car_weights !== nothing,
            car_precision = car_precision,
            sample_car_strength = sample_car_strength,
            car_strength_prior_a = car_strength_prior_a,
            car_strength_prior_b = car_strength_prior_b,
            car_degree_scale = car_degree_scale,
            car_kappa_init = car_kappa_init,
        ) :
        timed.value
    return (samples = samples, diagnostics = diagnostics)
end

function _prox_alpha_hyperparameters(prior::AbstractCoxPrior, alpha_prior_shape, alpha_prior_rate)
    shape = Float64(something(alpha_prior_shape, prior isa BayesianLassoPrior ? prior.lambda2_shape : 1.0))
    rate = Float64(something(alpha_prior_rate, prior isa BayesianLassoPrior ? prior.lambda2_rate : 1.0))
    _check_positive(shape, "alpha_prior_shape")
    _check_positive(rate, "alpha_prior_rate")
    return shape, rate
end

function _chains_value_array(chains)
    return Array(getfield(chains, :value))
end

function _chains_param_names(chains)
    return string.(names(chains))
end

function _prox_continuous_draws(chains, p::Integer)
    arr = _chains_value_array(chains)
    param_names = _chains_param_names(chains)
    beta_idx = [findfirst(==(string("beta[$j]")), param_names) for j in 1:p]
    any(isnothing, beta_idx) && throw(ArgumentError("NUTS output does not contain all beta parameters."))
    eta_idx = findfirst(==("eta"), param_names)
    eta_idx === nothing && throw(ArgumentError("NUTS output does not contain eta."))
    n_iter = size(arr, 1)
    n_chain = size(arr, 3)
    draws = Vector{Matrix{Float64}}(undef, n_chain)
    for c in 1:n_chain
        mat = Matrix{Float64}(undef, n_iter, p + 1)
        for j in 1:p
            mat[:, j] .= arr[:, beta_idx[j], c]
        end
        mat[:, p + 1] .= arr[:, eta_idx, c]
        draws[c] = mat
    end
    return draws
end

function _sample_lasso_prox_nuts(
    data::CoxPreparedData,
    prior::BayesianLassoPrior;
    n_samples::Integer,
    warmup::Integer,
    chains::Integer,
    target_accept::Real,
    rng::AbstractRNG,
    constraint::Symbol,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape,
    alpha_prior_rate,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
    progress::Bool,
    callback = nothing,
)
    p = size(data.z, 2)
    a_shape, a_rate = _prox_alpha_hyperparameters(prior, alpha_prior_shape, alpha_prior_rate)
    model = _lasso_prox_turing_model(
        data,
        prior,
        constraint,
        Float64(prox_sum0_lambda),
        Float64(prox_l1_lambda),
        a_shape,
        a_rate,
        Float64(prox_root_tol),
        Int(prox_root_maxiter),
    )
    sampler = Turing.NUTS(warmup, Float64(target_accept))
    timed = @timed if chains == 1
        Turing.sample(rng, model, sampler, n_samples; progress = progress, callback = callback)
    else
        Turing.sample(rng, model, sampler, Turing.MCMCThreads(), n_samples, chains; progress = progress, callback = callback)
    end
    chain_draws = _prox_continuous_draws(timed.value, p)
    names = vcat(["beta[$j]" for j in 1:p], ["eta"])
    samples = Vector{MHSamples}(undef, length(chain_draws))
    for c in eachindex(chain_draws)
        draws = chain_draws[c]
        lps = [_lasso_prox_logposterior(
            view(draws, i, :),
            data,
            prior;
            constraint = constraint,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = a_shape,
            alpha_prior_rate = a_rate,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
        ) for i in axes(draws, 1)]
        samples[c] = MHSamples(draws, lps, names, NaN, NaN, fill(NaN, p + 1))
    end
    gc_percent = timed.time > 0 ? 100 * timed.gctime / timed.time : NaN
    diagnostics = SamplingDiagnostics(missing, missing, timed.time, Int64(timed.bytes), timed.gctime, gc_percent)
    return (samples = chains == 1 ? samples[1] : MultiMHSamples(samples), diagnostics = diagnostics)
end

function _sample_ssl_prox_nuts_gibbs(
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior;
    n_samples::Integer,
    warmup::Integer,
    thin::Integer,
    chains::Integer,
    target_accept::Real,
    rng::AbstractRNG,
    constraint::Symbol,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape,
    alpha_prior_rate,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
    progress::Bool,
    progress_desc::AbstractString = "SSLASSO prox NUTS-Gibbs",
    progress_offset::Integer = 0,
)
    timed = @timed begin
        if chains == 1
            _sample_ssl_prox_nuts_gibbs_one(
                data,
                prior;
                n_samples = n_samples,
                warmup = warmup,
                thin = thin,
                rng = rng,
                target_accept = target_accept,
                constraint = constraint,
                prox_sum0_lambda = prox_sum0_lambda,
                prox_l1_lambda = prox_l1_lambda,
                alpha_prior_shape = alpha_prior_shape,
                alpha_prior_rate = alpha_prior_rate,
                prox_root_tol = prox_root_tol,
                prox_root_maxiter = prox_root_maxiter,
                progress = progress,
                progress_desc = progress_desc,
                progress_offset = progress_offset,
            )
        else
            samples = Vector{MHSamples}(undef, chains)
            rngs = [MersenneTwister(rand(rng, UInt64)) for _ in 1:chains]
            _reserve_chain_progress_lines(chains, progress)
            Threads.@threads for c in 1:chains
                samples[c] = _sample_ssl_prox_nuts_gibbs_one(
                    data,
                    prior;
                    n_samples = n_samples,
                    warmup = warmup,
                    thin = thin,
                    rng = rngs[c],
                    target_accept = target_accept,
                    constraint = constraint,
                    prox_sum0_lambda = prox_sum0_lambda,
                    prox_l1_lambda = prox_l1_lambda,
                    alpha_prior_shape = alpha_prior_shape,
                    alpha_prior_rate = alpha_prior_rate,
                    prox_root_tol = prox_root_tol,
                    prox_root_maxiter = prox_root_maxiter,
                    progress = progress,
                    progress_desc = _chain_progress_desc("SSLASSO prox NUTS-Gibbs", c, chains),
                    progress_offset = _chain_progress_offset(c, chains),
                )
            end
            MultiMHSamples(samples)
        end
    end
    gc_percent = timed.time > 0 ? 100 * timed.gctime / timed.time : NaN
    diagnostics = SamplingDiagnostics(missing, missing, timed.time, Int64(timed.bytes), timed.gctime, gc_percent)
    return (samples = timed.value, diagnostics = diagnostics)
end

function _sample_ssl_prox_nuts_gibbs_one(
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior;
    n_samples::Integer,
    warmup::Integer,
    thin::Integer,
    rng::AbstractRNG,
    target_accept::Real,
    constraint::Symbol,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape,
    alpha_prior_rate,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
    progress::Bool,
    progress_desc::AbstractString = "SSLASSO prox NUTS-Gibbs",
    progress_offset::Integer = 0,
)
    p = size(data.z, 2)
    a_shape, a_rate = _prox_alpha_hyperparameters(prior, alpha_prior_shape, alpha_prior_rate)
    gamma = trues(p)
    theta = 0.5
    total_iter = warmup + n_samples * thin
    names = vcat(["beta[$j]" for j in 1:p], ["eta"], ["gamma[$j]" for j in 1:p], ["theta"])
    draws = Matrix{Float64}(undef, n_samples, length(names))
    lps = Vector{Float64}(undef, n_samples)
    saved = 0
    prog = _progress_meter(progress, total_iter, "SSLASSO prox NUTS-Gibbs"; progress_desc = progress_desc, progress_offset = progress_offset)
    u = vcat(zeros(Float64, p), [0.0])
    for iter in 1:total_iter
        model = _ssl_prox_turing_block_model(
            data,
            prior,
            gamma,
            theta,
            constraint,
            Float64(prox_sum0_lambda),
            Float64(prox_l1_lambda),
            a_shape,
            a_rate,
            Float64(prox_root_tol),
            Int(prox_root_maxiter),
        )
        chain = Turing.sample(rng, model, Turing.NUTS(0, Float64(target_accept)), 1; progress = false)
        block_draw = _prox_continuous_draws(chain, p)[1]
        u = vec(block_draw[1, :])
        beta = view(u, 1:p)
        alpha = exp(clamp(u[p + 1], -700.0, 700.0))
        theta_clamped = clamp(theta, eps(Float64), 1 - eps(Float64))
        for j in 1:p
            gamma[j] = false
            h0 = weighted_l1_moreau(beta, alpha, _ssl_weights(prior, gamma), prox_l1_lambda;
                prox_root_tol = prox_root_tol, prox_root_maxiter = prox_root_maxiter)
            gamma[j] = true
            h1 = weighted_l1_moreau(beta, alpha, _ssl_weights(prior, gamma), prox_l1_lambda;
                prox_root_tol = prox_root_tol, prox_root_maxiter = prox_root_maxiter)
            logit_p = log(theta_clamped) - log1p(-theta_clamped) - (h1 - h0)
            gamma[j] = rand(rng) < _logistic(logit_p)
        end
        theta = rand(rng, Beta(prior.theta_a + count(gamma), prior.theta_b + p - count(gamma)))

        if iter > warmup && (iter - warmup) % thin == 0
            saved += 1
            @views draws[saved, 1:p] .= beta
            draws[saved, p + 1] = u[p + 1]
            @views draws[saved, (p + 2):(2p + 1)] .= Float64.(gamma)
            draws[saved, 2p + 2] = theta
            lps[saved] = _ssl_prox_logposterior(
                u,
                data,
                prior,
                gamma,
                theta;
                constraint = constraint,
                prox_sum0_lambda = prox_sum0_lambda,
                prox_l1_lambda = prox_l1_lambda,
                alpha_prior_shape = a_shape,
                alpha_prior_rate = a_rate,
                prox_root_tol = prox_root_tol,
                prox_root_maxiter = prox_root_maxiter,
            )
        end
        phase = iter <= warmup ? "warmup $(iter)/$(warmup)" : "sample $(saved)/$(n_samples)"
        _progress_next!(prog; showvalues = [(:phase, phase), (:saved, saved)])
    end
    _progress_finish!(prog)
    return MHSamples(draws, lps, names, NaN, NaN, fill(NaN, length(names)))
end
