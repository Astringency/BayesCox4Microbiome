"""
    sample_bayes_cox(X, time, event; prior=RegularizedHorseshoePrior(), sampler=:nuts, ...)

Fit the Bayesian Cox model for compositional microbiome covariates.

The default `sampler=:nuts` follows the PDF and uses Turing.jl's NUTS implementation.
Use `sampler=:hmc` for explicit-gradient AdvancedHMC with fixed leapfrog trajectories,
or `sampler=:mh` for the dependency-light adaptive random-walk Metropolis fallback.
For HS/RHS-family priors, pass `noncentered=true` with NUTS/HMC to sample latent
standard-normal `beta_raw` variables internally while returning the usual
`beta[...]` posterior draws.
"""
function _estimate_cox_pseudo_mu(
    data::CoxPreparedData;
    tol::Real = 1e-4,
    l2_cost::Real = 1.0,
)
    event_time = Survival.EventTime.(data.time, data.event .!= 0)
    fit = Survival.coxph(data.z, event_time; tol = tol, l2_cost = l2_cost)
    beta = Survival.coef(fit)
    eta = clamp.(data.z * beta, -30.0, 30.0)
    return mean(exp.(eta))
end

function _p0_prior_name(name)
    key = lowercase(replace(string(name), "_" => "-", " " => "-"))
    return key in (
        "hs",
        "rhs",
        "new-rhs",
        "est-rhs",
        "cox-hs",
        "cox-rhs",
    )
end

function _clamp_p0_estimate(count_sig::Integer, p::Integer; min_p0::Integer = 1, max_p0 = nothing)
    p > 1 || throw(ArgumentError("p must be larger than one to estimate p0."))
    lo = clamp(min_p0, 1, p - 1)
    hi = max_p0 === nothing ? p - 1 : clamp(Int(max_p0), lo, p - 1)
    return clamp(count_sig, lo, hi)
end

"""
    estimate_p0_univariate_cox(X, time, event; alpha=0.05, ...)

Estimate the prior guess `p0` for HS/RHS-style priors by fitting one
univariate Cox PH model per covariate with `Survival.jl`. The estimate is the
number of covariates whose Wald test p-value is below `alpha`, clamped to
`[min_p0, max_p0]` and to be strictly smaller than `p`.
"""
function estimate_p0_univariate_cox(
    data::CoxPreparedData;
    alpha::Real = 0.05,
    min_p0::Integer = 1,
    max_p0 = nothing,
    tol::Real = 1e-4,
    l2_cost::Real = 0.0,
)
    0 < alpha < 1 || throw(ArgumentError("alpha must be in (0, 1)."))
    p = size(data.z, 2)
    event_time = Survival.EventTime.(data.time, data.event .!= 0)
    estimate = fill(NaN, p)
    stderr = fill(NaN, p)
    zscore = fill(NaN, p)
    pvalue = fill(1.0, p)
    converged = falses(p)
    significant = falses(p)

    for j in 1:p
        xj = view(data.z, :, j)
        if std(xj) <= eps(Float64)
            continue
        end
        try
            fit = Survival.coxph(reshape(collect(xj), :, 1), event_time; tol = tol, l2_cost = l2_cost)
            b = Survival.coef(fit)[1]
            se = Survival.stderror(fit)[1]
            if isfinite(b) && isfinite(se) && se > 0
                z = b / se
                pv = 2 * cdf(Normal(), -abs(z))
                estimate[j] = b
                stderr[j] = se
                zscore[j] = z
                pvalue[j] = pv
                converged[j] = true
                significant[j] = pv < alpha
            end
        catch
            pvalue[j] = 1.0
        end
    end

    p0 = _clamp_p0_estimate(count(significant), p; min_p0 = min_p0, max_p0 = max_p0)
    return (
        p0 = p0,
        alpha = Float64(alpha),
        significant = significant,
        pvalue = pvalue,
        estimate = estimate,
        stderr = stderr,
        zscore = zscore,
        converged = converged,
    )
end

function estimate_p0_univariate_cox(
    X::AbstractMatrix{<:Real},
    time::AbstractVector{<:Real},
    event::AbstractVector;
    transform::Symbol = :log,
    pseudocount::Real = 1e-6,
    closure::Bool = true,
    standardize::Bool = false,
    kwargs...,
)
    data = prepare_cox_data(
        X,
        time,
        event;
        transform = transform,
        pseudocount = pseudocount,
        closure = closure,
        standardize = standardize,
    )
    return estimate_p0_univariate_cox(data; kwargs...)
end

function _resolve_p0(
    name,
    p0,
    X,
    time,
    event;
    transform::Symbol = :log,
    pseudocount::Real = 1e-6,
    closure::Bool = true,
    p0_alpha::Real = 0.05,
    p0_min::Integer = 1,
    p0_max = nothing,
)
    p0 !== nothing && return Float64(p0)
    _p0_prior_name(name) ||
        return 1.0
    (X === nothing || time === nothing || event === nothing) &&
        throw(ArgumentError("p0 was not provided; X, time, and event are required for univariate Cox screening."))
    screen = estimate_p0_univariate_cox(
        X,
        time,
        event;
        transform = transform,
        pseudocount = pseudocount,
        closure = closure,
        alpha = p0_alpha,
        min_p0 = p0_min,
        max_p0 = p0_max,
    )
    return Float64(screen.p0)
end

"""
    cox_paper_prior(name; p0=nothing, X=nothing, time=nothing, event=nothing, realdata=false)

Construct one of the prior specifications used in the simulation section
of `docs/cox4microbiome_detail.pdf`: `hs`, `rhs`, `new-rhs`, `est-rhs`,
`cox-hs`, `cox-rhs`, `lasso`, and `sslasso`. The additional names
`prox-lasso` and `prox-sslasso` select the proximal L1 implementations.

When `p0` is `nothing`, HS/RHS-style priors estimate `p0` by univariate Cox
screening and use the number of covariates significant at `p0_alpha`. RHS
slab, BLASSO shrinkage, and SSLASSO spike/slab hyperparameters can be tuned
with the `rhs_*`, `lasso_*`, and `sslasso_*` keyword arguments.
"""
function cox_paper_prior(
    name::Union{AbstractString,Symbol};
    p0 = nothing,
    X = nothing,
    time = nothing,
    event = nothing,
    realdata::Bool = false,
    transform::Symbol = :log,
    pseudocount::Real = 1e-6,
    closure::Bool = true,
    cox_estimate_tol::Real = 1e-4,
    cox_estimate_l2_cost::Real = 1.0,
    p0_alpha::Real = 0.05,
    p0_min::Integer = 1,
    p0_max = nothing,
    rhs_slab_df::Real = 4.0,
    rhs_slab_s2::Real = 2.0,
    hs_tau_scale::Real = 1.0,
    hs_local_scale::Real = 1.0,
    rhs_tau_scale::Real = 1.0,
    rhs_local_scale::Real = 1.0,
    rhs_pseudo_sigma_df::Real = 3.0,
    rhs_pseudo_sigma_scale::Real = 10.0,
    lasso_lambda = nothing,
    lasso_lambda2_shape::Real = 1.0,
    lasso_lambda2_rate::Real = 0.1,
    sslasso_spike_scale = nothing,
    sslasso_slab_scale = nothing,
    sslasso_theta_a::Real = 1.0,
    sslasso_theta_b::Real = 1.0,
)
    key = lowercase(replace(string(name), "_" => "-", " " => "-"))
    rhs_df = Float64(rhs_slab_df)
    rhs_s2 = Float64(rhs_slab_s2)
    lasso_lambda_value = lasso_lambda === nothing ? nothing : Float64(lasso_lambda)
    ssl_spike_default = realdata ? 1.0 : 0.04
    ssl_slab_default = realdata ? 10.0 : 0.5
    ssl_spike = Float64(something(sslasso_spike_scale, ssl_spike_default))
    ssl_slab = Float64(something(sslasso_slab_scale, ssl_slab_default))
    p0_value = _resolve_p0(
        name,
        p0,
        X,
        time,
        event;
        transform = transform,
        pseudocount = pseudocount,
        closure = closure,
        p0_alpha = p0_alpha,
        p0_min = p0_min,
        p0_max = p0_max,
    )
    if key == "hs"
        return HorseshoePrior(
            tau_scale = Float64(hs_tau_scale),
            use_cox_tau0 = false,
            p0 = p0_value,
            local_scale = Float64(hs_local_scale),
        )
    elseif key == "rhs"
        return RegularizedHorseshoePrior(
            tau_scale = Float64(rhs_tau_scale),
            use_cox_tau0 = false,
            p0 = p0_value,
            local_scale = Float64(rhs_local_scale),
            tau_formula = :linear,
            pseudo_sigma = nothing,
            sample_pseudo_sigma = true,
            pseudo_sigma_df = Float64(rhs_pseudo_sigma_df),
            pseudo_sigma_scale = Float64(rhs_pseudo_sigma_scale),
            slab_df = rhs_df,
            slab_s2 = rhs_s2,
        )
    elseif key == "new-rhs"
        return RegularizedHorseshoePrior(
            tau_scale = Float64(rhs_tau_scale),
            use_cox_tau0 = false,
            p0 = p0_value,
            local_scale = Float64(rhs_local_scale),
            tau_formula = :linear,
            pseudo_sigma = 1.0,
            slab_df = rhs_df,
            slab_s2 = rhs_s2,
        )
    elseif key == "est-rhs"
        (X === nothing || time === nothing || event === nothing) &&
            throw(ArgumentError("est-rhs requires X, time, and event to estimate pseudo_sigma."))
        data = prepare_cox_data(
            X,
            time,
            event;
            transform = transform,
            pseudocount = pseudocount,
            closure = closure,
            standardize = false,
        )
        muhat = _estimate_cox_pseudo_mu(
            data;
            tol = cox_estimate_tol,
            l2_cost = cox_estimate_l2_cost,
        )
        return RegularizedHorseshoePrior(
            tau_scale = Float64(rhs_tau_scale),
            use_cox_tau0 = false,
            p0 = p0_value,
            local_scale = Float64(rhs_local_scale),
            tau_formula = :linear,
            pseudo_sigma = sqrt(inv(max(muhat, eps(Float64)))),
            slab_df = rhs_df,
            slab_s2 = rhs_s2,
        )
    elseif key == "cox-hs"
        return HorseshoePrior(
            tau_scale = Float64(hs_tau_scale),
            use_cox_tau0 = true,
            p0 = p0_value,
            local_scale = Float64(hs_local_scale),
        )
    elseif key == "cox-rhs"
        return RegularizedHorseshoePrior(
            tau_scale = Float64(rhs_tau_scale),
            use_cox_tau0 = true,
            p0 = p0_value,
            local_scale = Float64(rhs_local_scale),
            tau_formula = :cox,
            slab_df = rhs_df,
            slab_s2 = rhs_s2,
        )
    elseif key == "lasso"
        return BayesianLassoPrior(
            lambda = lasso_lambda_value,
            lambda2_shape = Float64(lasso_lambda2_shape),
            lambda2_rate = Float64(lasso_lambda2_rate),
        )
    elseif key == "prox-lasso"
        return BayesianLassoPrior(
            lambda = lasso_lambda_value,
            lambda2_shape = Float64(lasso_lambda2_shape),
            lambda2_rate = Float64(lasso_lambda2_rate),
            proximal = true,
        )
    elseif key == "sslasso"
        return SpikeSlabLassoPrior(
            spike_scale = ssl_spike,
            slab_scale = ssl_slab,
            theta_a = Float64(sslasso_theta_a),
            theta_b = Float64(sslasso_theta_b),
        )
    elseif key == "prox-sslasso"
        return SpikeSlabLassoPrior(
            spike_scale = ssl_spike,
            slab_scale = ssl_slab,
            theta_a = Float64(sslasso_theta_a),
            theta_b = Float64(sslasso_theta_b),
            proximal = true,
        )
    end
    throw(ArgumentError("unsupported paper prior name: $name"))
end

function _validate_prox_prior_sampler(prior::AbstractCoxPrior, sampler::Symbol)
    _is_proximal_l1_prior(prior) || return nothing
    if sampler == :mh
        throw(ArgumentError(
            "proximal L1 priors are not implemented for sampler=:mh; use :nuts, :hmc, :mala, :ula, " *
            "or :gibbs for prox-sslasso.",
        ))
    elseif sampler == :gibbs && !(prior isa SpikeSlabLassoPrior)
        throw(ArgumentError("sampler=:gibbs only supports the proximal L1 path for prox-sslasso."))
    end
    return nothing
end

function sample_bayes_cox(
    X::AbstractMatrix{<:Real},
    time::AbstractVector{<:Real},
    event::AbstractVector;
    prior::AbstractCoxPrior = RegularizedHorseshoePrior(),
    sampler::Symbol = :nuts,
    nuts_backend::Symbol = :turing,
    transform::Symbol = :log,
    pseudocount::Real = 1e-6,
    closure::Bool = true,
    standardize::Bool = false,
    tie_method::Symbol = :efron,
    use_car::Bool = false,
    car_weights = nothing,
    car_distance = nothing,
    car_weight_quantile::Real = 0.75,
    car_precision::Real = 1.0,
    sample_car_strength::Bool = false,
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_degree_scale::Symbol = :mean_active,
    car_kappa_init::Real = 0.5,
    noncentered::Bool = false,
    kwargs...,
)
    car_weights !== nothing && car_distance !== nothing && use_car &&
        throw(ArgumentError("pass either car_weights or car_distance, not both."))
    resolved_car_weights =
        if !use_car
            nothing
        elseif car_weights !== nothing
            _prepare_car_weights(car_weights, size(X, 2))
        else
            _car_weight_edges(car_weight_matrix(
                X;
                distance = car_distance === nothing ? :bray : car_distance,
                threshold_quantile = car_weight_quantile,
            ))
        end
    data = prepare_cox_data(
        X,
        time,
        event;
        transform = transform,
        pseudocount = pseudocount,
        closure = closure,
        standardize = standardize,
        tie_method = tie_method,
    )
    return sample_bayes_cox(
        data;
        prior = prior,
        sampler = sampler,
        nuts_backend = nuts_backend,
        use_car = use_car,
        car_weights = resolved_car_weights,
        car_precision = car_precision,
        sample_car_strength = sample_car_strength,
        car_strength_prior_a = car_strength_prior_a,
        car_strength_prior_b = car_strength_prior_b,
        car_degree_scale = car_degree_scale,
        car_kappa_init = car_kappa_init,
        noncentered = noncentered,
        kwargs...,
    )
end

function sample_bayes_cox(
    data::CoxPreparedData;
    prior::AbstractCoxPrior = RegularizedHorseshoePrior(),
    sampler::Symbol = :nuts,
    nuts_backend::Symbol = :turing,
    seed = nothing,
    chains = nothing,
    multichains::Bool = false,
    use_car::Bool = false,
    car_weights = nothing,
    car_distance = nothing,
    car_weight_quantile::Real = 0.75,
    car_precision::Real = 1.0,
    sample_car_strength::Bool = false,
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_degree_scale::Symbol = :mean_active,
    car_kappa_init::Real = 0.5,
    noncentered::Bool = false,
    kwargs...,
)
    car_weights !== nothing && car_distance !== nothing && use_car &&
        throw(ArgumentError("pass either car_weights or car_distance, not both."))
    resolved_car_weights =
        if !use_car
            nothing
        elseif car_weights !== nothing
            _prepare_car_weights(car_weights, size(data.z, 2))
        else
            data.transform == :log && !data.standardize ||
                throw(ArgumentError(
                    "car_distance for CoxPreparedData requires log-transformed, unstandardized data. " *
                    "Pass car_weights explicitly or call sample_bayes_cox with the original X matrix.",
                ))
            _car_weight_edges(car_weight_matrix(
                exp.(data.z);
                distance = car_distance === nothing ? :bray : car_distance,
                threshold_quantile = car_weight_quantile,
            ))
        end
    if sample_car_strength
        prior isa Union{HorseshoePrior,RegularizedHorseshoePrior} ||
            throw(ArgumentError("sample_car_strength requires a Horseshoe or RegularizedHorseshoe prior with CAR-on-log-lambda local scales."))
        _validate_adaptive_car_options(
            resolved_car_weights,
            car_strength_prior_a,
            car_strength_prior_b,
            car_kappa_init,
        )
    end
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    effective_sampler =
        sampler in (:nuts_explicit, :nuts_turing) ? :nuts : sampler
    effective_nuts_backend =
        sampler == :nuts_explicit ? :explicit :
        sampler == :nuts_turing ? :turing : nuts_backend
    effective_nuts_backend in (:turing, :explicit) ||
        throw(ArgumentError("nuts_backend must be :turing or :explicit."))
    _validate_prox_prior_sampler(prior, effective_sampler)
    chain_count = chains === nothing ? (effective_sampler == :nuts ? nothing : 1) : Int(chains)
    chain_count !== nothing && chain_count <= 0 && throw(ArgumentError("chains must be positive."))
    if effective_sampler == :nuts
        sampled =
            if effective_nuts_backend == :turing
                if chain_count === nothing
                    _sample_nuts(
                        data,
                        prior;
                        rng = rng,
                        car_weights = resolved_car_weights,
                        use_car = use_car,
                        car_precision = car_precision,
                        sample_car_strength = sample_car_strength,
                        car_strength_prior_a = car_strength_prior_a,
                        car_strength_prior_b = car_strength_prior_b,
                        car_degree_scale = car_degree_scale,
                        car_kappa_init = car_kappa_init,
                        noncentered = noncentered,
                        kwargs...,
                    )
                else
                    _sample_nuts(
                        data,
                        prior;
                        rng = rng,
                        chains = chain_count,
                        car_weights = resolved_car_weights,
                        use_car = use_car,
                        car_precision = car_precision,
                        sample_car_strength = sample_car_strength,
                        car_strength_prior_a = car_strength_prior_a,
                        car_strength_prior_b = car_strength_prior_b,
                        car_degree_scale = car_degree_scale,
                        car_kappa_init = car_kappa_init,
                        noncentered = noncentered,
                        kwargs...,
                    )
                end
            else
                if chain_count === nothing
                    _sample_nuts_explicit(
                        data,
                        prior;
                        rng = rng,
                        multichains = multichains,
                        car_weights = resolved_car_weights,
                        use_car = use_car,
                        car_precision = car_precision,
                        sample_car_strength = sample_car_strength,
                        car_strength_prior_a = car_strength_prior_a,
                        car_strength_prior_b = car_strength_prior_b,
                        car_degree_scale = car_degree_scale,
                        car_kappa_init = car_kappa_init,
                        noncentered = noncentered,
                        kwargs...,
                    )
                else
                    _sample_nuts_explicit(
                        data,
                        prior;
                        rng = rng,
                        chains = chain_count,
                        multichains = multichains,
                        car_weights = resolved_car_weights,
                        use_car = use_car,
                        car_precision = car_precision,
                        sample_car_strength = sample_car_strength,
                        car_strength_prior_a = car_strength_prior_a,
                        car_strength_prior_b = car_strength_prior_b,
                        car_degree_scale = car_degree_scale,
                        car_kappa_init = car_kappa_init,
                        noncentered = noncentered,
                        kwargs...,
                    )
                end
            end
        samples = sampled.samples
        diagnostics = sampled.diagnostics
    elseif effective_sampler == :hmc
        sampled = _sample_hmc_explicit(
            data,
            prior;
            rng = rng,
            chains = chain_count,
            multichains = multichains,
            car_weights = resolved_car_weights,
            use_car = use_car,
            car_precision = car_precision,
            sample_car_strength = sample_car_strength,
            car_strength_prior_a = car_strength_prior_a,
            car_strength_prior_b = car_strength_prior_b,
            car_degree_scale = car_degree_scale,
            car_kappa_init = car_kappa_init,
            noncentered = noncentered,
            kwargs...,
        )
        samples = sampled.samples
        diagnostics = sampled.diagnostics
    elseif effective_sampler == :mh
        noncentered && throw(ArgumentError("noncentered=true is currently supported only for sampler=:nuts or :hmc."))
        timed = @timed _sample_mh_chains(
            data,
            prior,
            chain_count;
            rng = rng,
            multichains = multichains,
            car_weights = resolved_car_weights,
            use_car = use_car,
            car_precision = car_precision,
            sample_car_strength = sample_car_strength,
            car_strength_prior_a = car_strength_prior_a,
            car_strength_prior_b = car_strength_prior_b,
            car_degree_scale = car_degree_scale,
            car_kappa_init = car_kappa_init,
            kwargs...,
        )
        samples = timed.value
        warmup_seconds = missing
        sample_seconds = missing
        gc_percent = timed.time > 0 ? 100 * timed.gctime / timed.time : NaN
        diagnostics = SamplingDiagnostics(
            warmup_seconds,
            sample_seconds,
            timed.time,
            Int64(timed.bytes),
            timed.gctime,
            gc_percent,
        )
    elseif effective_sampler == :mala
        noncentered && throw(ArgumentError("noncentered=true is currently supported only for sampler=:nuts or :hmc."))
        timed = @timed _sample_mala_chains(
            data,
            prior,
            chain_count;
            rng = rng,
            multichains = multichains,
            car_weights = resolved_car_weights,
            use_car = use_car,
            car_precision = car_precision,
            sample_car_strength = sample_car_strength,
            car_strength_prior_a = car_strength_prior_a,
            car_strength_prior_b = car_strength_prior_b,
            car_degree_scale = car_degree_scale,
            car_kappa_init = car_kappa_init,
            kwargs...,
        )
        samples = timed.value
        warmup_seconds = missing
        sample_seconds = missing
        gc_percent = timed.time > 0 ? 100 * timed.gctime / timed.time : NaN
        diagnostics = SamplingDiagnostics(
            warmup_seconds,
            sample_seconds,
            timed.time,
            Int64(timed.bytes),
            timed.gctime,
            gc_percent,
        )
    elseif effective_sampler == :ula
        noncentered && throw(ArgumentError("noncentered=true is currently supported only for sampler=:nuts or :hmc."))
        timed = @timed _sample_ula_chains(
            data,
            prior,
            chain_count;
            rng = rng,
            multichains = multichains,
            car_weights = resolved_car_weights,
            use_car = use_car,
            car_precision = car_precision,
            sample_car_strength = sample_car_strength,
            car_strength_prior_a = car_strength_prior_a,
            car_strength_prior_b = car_strength_prior_b,
            car_degree_scale = car_degree_scale,
            car_kappa_init = car_kappa_init,
            kwargs...,
        )
        samples = timed.value
        warmup_seconds = missing
        sample_seconds = missing
        gc_percent = timed.time > 0 ? 100 * timed.gctime / timed.time : NaN
        diagnostics = SamplingDiagnostics(
            warmup_seconds,
            sample_seconds,
            timed.time,
            Int64(timed.bytes),
            timed.gctime,
            gc_percent,
        )
    elseif effective_sampler == :gibbs
        noncentered && throw(ArgumentError("noncentered=true is currently supported only for sampler=:nuts or :hmc."))
        prior isa SpikeSlabLassoPrior && use_car && resolved_car_weights !== nothing &&
            throw(ArgumentError("sampler=:gibbs for SSLASSO does not support CAR weights."))
        timed =
            if prior isa SpikeSlabLassoPrior
                @timed _sample_gibbs_chains(
                    data,
                    prior,
                    chain_count;
                    rng = rng,
                    multichains = multichains,
                    kwargs...,
                )
            else
                @timed _sample_gibbs_chains(
                    data,
                    prior,
                    chain_count;
                    rng = rng,
                    multichains = multichains,
                    car_weights = resolved_car_weights,
                    use_car = use_car,
                    car_precision = car_precision,
                    sample_car_strength = sample_car_strength,
                    car_strength_prior_a = car_strength_prior_a,
                    car_strength_prior_b = car_strength_prior_b,
                    car_degree_scale = car_degree_scale,
                    car_kappa_init = car_kappa_init,
                    kwargs...,
                )
            end
        samples = timed.value
        warmup_seconds = missing
        sample_seconds = missing
        gc_percent = timed.time > 0 ? 100 * timed.gctime / timed.time : NaN
        diagnostics = SamplingDiagnostics(
            warmup_seconds,
            sample_seconds,
            timed.time,
            Int64(timed.bytes),
            timed.gctime,
            gc_percent,
        )
    else
        throw(ArgumentError("sampler must be :nuts, :nuts_turing, :nuts_explicit, :hmc, :mh, :mala, :ula, or :gibbs."))
    end
    return CoxSamplerResult(samples, data, prior, effective_sampler, diagnostics)
end

sampler_diagnostics(result::CoxSamplerResult) = result.diagnostics

function _mhsamples_to_chains(samples::Vector{MHSamples})
    isempty(samples) && throw(ArgumentError("at least one chain is required."))
    names_with_lp = vcat(samples[1].names, ["logposterior"])
    n_iter = size(samples[1].draws, 1)
    n_param = length(names_with_lp)
    n_chain = length(samples)
    arr = Array{Float64}(undef, n_iter, n_param, n_chain)
    for (c, sample) in enumerate(samples)
        sample.names == samples[1].names ||
            throw(ArgumentError("all chains must have identical parameter names."))
        size(sample.draws, 1) == n_iter ||
            throw(ArgumentError("all chains must have the same number of saved draws."))
        size(sample.draws, 2) + 1 == n_param ||
            throw(ArgumentError("sample draw dimension does not match parameter names."))
        arr[:, 1:(n_param - 1), c] .= sample.draws
        arr[:, n_param, c] .= sample.logposterior
    end
    MCMCChains = Base.require(Base.PkgId(Base.UUID("c7f686f2-ff18-58e9-bc7b-31028e88f75d"), "MCMCChains"))
    return Base.invokelatest(getproperty(MCMCChains, :Chains), arr, Symbol.(names_with_lp))
end

_mhsamples_to_chains(samples::MultiMHSamples) = _mhsamples_to_chains(samples.chains)

function _independent_rng(rng::AbstractRNG)
    return MersenneTwister(rand(rng, UInt64))
end

function _sample_mh_chains(
    data::CoxPreparedData,
    prior::AbstractCoxPrior,
    chains::Integer;
    rng::AbstractRNG,
    multichains::Bool = false,
    progress::Bool = true,
    kwargs...,
)
    chains == 1 && return _sample_mh(data, prior; rng = rng, progress = progress, kwargs...)
    samples = Vector{MHSamples}(undef, chains)
    rngs = [_independent_rng(rng) for _ in 1:chains]
    if multichains
        _reserve_chain_progress_lines(chains, progress)
        Threads.@threads for c in 1:chains
            @info "Running MH chain $c/$chains"
            samples[c] = _sample_mh(
                data,
                prior;
                rng = rngs[c],
                progress = progress,
                progress_desc = _chain_progress_desc("MH sampling", c, chains),
                progress_offset = _chain_progress_offset(c, chains),
                kwargs...,
            )
        end
    else
        for c in 1:chains
            @info "Running MH chain $c/$chains"
            samples[c] = _sample_mh(data, prior; rng = rngs[c], progress = progress, kwargs...)
        end
    end
    return MultiMHSamples(samples)
end

function _sample_mala_chains(
    data::CoxPreparedData,
    prior::AbstractCoxPrior,
    chains::Integer;
    rng::AbstractRNG,
    multichains::Bool = false,
    progress::Bool = true,
    kwargs...,
)
    chains == 1 && return _sample_mala(data, prior; rng = rng, progress = progress, kwargs...)
    samples = Vector{MHSamples}(undef, chains)
    rngs = [_independent_rng(rng) for _ in 1:chains]
    if multichains
        _reserve_chain_progress_lines(chains, progress)
        Threads.@threads for c in 1:chains
            @info "Running MALA chain $c/$chains"
            samples[c] = _sample_mala(
                data,
                prior;
                rng = rngs[c],
                progress = progress,
                progress_desc = _chain_progress_desc("MALA sampling", c, chains),
                progress_offset = _chain_progress_offset(c, chains),
                kwargs...,
            )
        end
    else
        for c in 1:chains
            @info "Running MALA chain $c/$chains"
            samples[c] = _sample_mala(data, prior; rng = rngs[c], progress = progress, kwargs...)
        end
    end
    return MultiMHSamples(samples)
end

function _sample_ula_chains(
    data::CoxPreparedData,
    prior::AbstractCoxPrior,
    chains::Integer;
    rng::AbstractRNG,
    multichains::Bool = false,
    progress::Bool = true,
    kwargs...,
)
    chains == 1 && return _sample_ula(data, prior; rng = rng, progress = progress, kwargs...)
    samples = Vector{MHSamples}(undef, chains)
    rngs = [_independent_rng(rng) for _ in 1:chains]
    if multichains
        _reserve_chain_progress_lines(chains, progress)
        Threads.@threads for c in 1:chains
            @info "Running ULA chain $c/$chains"
            samples[c] = _sample_ula(
                data,
                prior;
                rng = rngs[c],
                progress = progress,
                progress_desc = _chain_progress_desc("ULA sampling", c, chains),
                progress_offset = _chain_progress_offset(c, chains),
                kwargs...,
            )
        end
    else
        for c in 1:chains
            @info "Running ULA chain $c/$chains"
            samples[c] = _sample_ula(data, prior; rng = rngs[c], progress = progress, kwargs...)
        end
    end
    return MultiMHSamples(samples)
end

function _sample_gibbs_chains(
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior,
    chains::Integer;
    rng::AbstractRNG,
    multichains::Bool = false,
    progress::Bool = true,
    n_samples::Integer = 5000,
    warmup::Integer = 2000,
    thin::Integer = 1,
    target_accept::Real = 0.8,
    constraint::Symbol = :soft_center,
    prox_sum0_lambda::Real = 1e-3,
    prox_l1_lambda::Real = 1e-3,
    alpha_prior_shape = nothing,
    alpha_prior_rate = nothing,
    prox_root_tol::Real = 1e-10,
    prox_root_maxiter::Integer = 100,
    kwargs...,
)
    resolved_constraint = _resolve_constraint(constraint)
    if _is_proximal_l1_prior(prior)
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
            kwargs...,
        ).samples
    end
    chains == 1 && return _sample_ssl_gibbs(data, prior; rng = rng, chains = 1, progress = progress, n_samples = n_samples, warmup = warmup, thin = thin, target_accept = target_accept, constraint = resolved_constraint, prox_sum0_lambda = prox_sum0_lambda, kwargs...)
    samples = Vector{MHSamples}(undef, chains)
    rngs = [_independent_rng(rng) for _ in 1:chains]
    if multichains
        _reserve_chain_progress_lines(chains, progress)
        Threads.@threads for c in 1:chains
            @info "Running Gibbs chain $c/$chains"
            samples[c] = _sample_ssl_gibbs(
                data,
                prior;
                rng = rngs[c],
                chains = 1,
                progress = progress,
                progress_desc = _chain_progress_desc("Gibbs sampling", c, chains),
                progress_offset = _chain_progress_offset(c, chains),
                n_samples = n_samples,
                warmup = warmup,
                thin = thin,
                target_accept = target_accept,
                constraint = resolved_constraint,
                prox_sum0_lambda = prox_sum0_lambda,
                kwargs...,
            )
        end
    else
        for c in 1:chains
            @info "Running Gibbs chain $c/$chains"
            samples[c] = _sample_ssl_gibbs(data, prior; rng = rngs[c], chains = 1, progress = progress, n_samples = n_samples, warmup = warmup, thin = thin, target_accept = target_accept, constraint = resolved_constraint, prox_sum0_lambda = prox_sum0_lambda, kwargs...)
        end
    end
    return MultiMHSamples(samples)
end

function _sample_gibbs_chains(
    data::CoxPreparedData,
    prior::AbstractCoxPrior,
    chains::Integer;
    rng::AbstractRNG,
    multichains::Bool = false,
    progress::Bool = true,
    kwargs...,
)
    chains == 1 && return _sample_coordinate_gibbs(data, prior; rng = rng, progress = progress, kwargs...)
    samples = Vector{MHSamples}(undef, chains)
    rngs = [_independent_rng(rng) for _ in 1:chains]
    if multichains
        _reserve_chain_progress_lines(chains, progress)
        Threads.@threads for c in 1:chains
            @info "Running coordinate Gibbs chain $c/$chains"
            samples[c] = _sample_coordinate_gibbs(
                data,
                prior;
                rng = rngs[c],
                progress = progress,
                progress_desc = _chain_progress_desc("Coordinate Gibbs sampling", c, chains),
                progress_offset = _chain_progress_offset(c, chains),
                kwargs...,
            )
        end
    else
        for c in 1:chains
            @info "Running coordinate Gibbs chain $c/$chains"
            samples[c] = _sample_coordinate_gibbs(data, prior; rng = rngs[c], progress = progress, kwargs...)
        end
    end
    return MultiMHSamples(samples)
end

function posterior_beta_draws(result::CoxSamplerResult)
    beta_indices(names, p::Integer) = begin
        sample_names = string.(names)
        idx = [findfirst(==("beta[$j]"), sample_names) for j in 1:p]
        if any(isnothing, idx)
            missing = ["beta[$j]" for j in 1:p if idx[j] === nothing]
            throw(ArgumentError(
                "sampler result is missing beta columns: $(join(missing, ", ")).",
            ))
        end
        return Int.(idx)
    end
    if result.samples isa MHSamples
        p = size(result.data.z, 2)
        idx = beta_indices(result.samples.names, p)
        return Matrix{Float64}(result.samples.draws[:, idx])
    elseif result.samples isa MultiMHSamples
        p = size(result.data.z, 2)
        return reduce(vcat, begin
            idx = beta_indices(sample.names, p)
            Matrix{Float64}(sample.draws[:, idx])
        end for sample in result.samples.chains)
    end
    p = size(result.data.z, 2)
    beta_vars = [Symbol("beta[$j]") for j in 1:p]
    try
        subchain = Base.invokelatest(getindex, result.samples, :, beta_vars, :)
        arr = Base.invokelatest(Array, subchain)
        if ndims(arr) == 2
            return Matrix{Float64}(arr)
        elseif ndims(arr) == 3
            n_iter, n_beta, n_chain = size(arr)
            n_beta == p || throw(DimensionMismatch("extracted beta dimension does not match data."))
            return reshape(permutedims(arr, (1, 3, 2)), n_iter * n_chain, n_beta)
        end
    catch err
        throw(ArgumentError(
            "could not extract beta variables from sampler result. " *
            "Expected variables named beta[1], ..., beta[$p]. Original error type: $(typeof(err))",
        ))
    end
    throw(ArgumentError("unsupported sampler result array shape for beta extraction."))
end

function _chain_variable_draws(chain, vars)
    all_vars = Base.invokelatest(names, chain)
    idx = map(vars) do var
        found = findfirst(==(var), all_vars)
        found === nothing && throw(ArgumentError("variable $var was not found in chain."))
        return found
    end
    chain_value = getfield(chain, :value)
    arr = Base.invokelatest(Array, chain_value)
    if ndims(arr) == 2
        return Matrix{Float64}(arr[:, idx])
    elseif ndims(arr) == 3
        n_iter, _, n_chain = size(arr)
        selected = arr[:, idx, :]
        return reshape(permutedims(selected, (1, 3, 2)), n_iter * n_chain, length(idx))
    end
    throw(ArgumentError("unsupported sampler result array shape."))
end

function posterior_logposterior_draws(result::CoxSamplerResult)
    if result.samples isa MHSamples
        return result.samples.logposterior
    elseif result.samples isa MultiMHSamples
        return reduce(vcat, [sample.logposterior for sample in result.samples.chains])
    end
    for stat in (:logposterior, :logjoint, :log_density, :lp)
        try
            values = _chain_variable_draws(result.samples, [stat])
            return vec(values[:, 1])
        catch
        end
    end
    throw(ArgumentError(
        "could not extract posterior log density from sampler result; " *
        "expected a logjoint, log_density, or lp variable.",
    ))
end

function posterior_beta_map(result::CoxSamplerResult)
    draws = posterior_beta_draws(result)
    logpost = posterior_logposterior_draws(result)
    length(logpost) == size(draws, 1) ||
        throw(DimensionMismatch("posterior log density count does not match beta draw count."))
    idx = argmax(logpost)
    return vec(draws[idx, :])
end

function posterior_beta_summary(result::CoxSamplerResult; cred_mass::Real = 0.95)
    0 < cred_mass < 1 || throw(ArgumentError("cred_mass must be in (0, 1)."))
    draws = posterior_beta_draws(result)
    p = size(draws, 2)
    alpha = (1 - cred_mass) / 2
    beta_mean = vec(mean(draws; dims=1))
    beta_median = [median(view(draws, :, j)) for j in 1:p]
    beta_map = posterior_beta_map(result)
    lower = [quantile(view(draws, :, j), alpha) for j in 1:p]
    upper = [quantile(view(draws, :, j), 1 - alpha) for j in 1:p]
    return (mean = beta_mean, median = beta_median, map = beta_map, lower = lower, upper = upper)
end

function sslasso_inclusion_probabilities(result::CoxSamplerResult)
    result.prior isa SpikeSlabLassoPrior ||
        throw(ArgumentError("sslasso_inclusion_probabilities requires SpikeSlabLassoPrior."))

    prior = result.prior
    p = size(result.data.z, 2)
    gamma_names = ["gamma[$j]" for j in 1:p]
    if result.samples isa MHSamples || result.samples isa MultiMHSamples
        sample_list = result.samples isa MHSamples ? [result.samples] : result.samples.chains
        names0 = sample_list[1].names
        draws = reduce(vcat, [sample.draws for sample in sample_list])
        gamma_idx = [findfirst(==(name), names0) for name in gamma_names]
        if all(!isnothing, gamma_idx)
            return vec(mean(draws[:, Int.(gamma_idx)]; dims = 1))
        end
        theta_idx = findfirst(==("logit_theta"), names0)
        theta_idx === nothing && (theta_idx = findfirst(==("theta"), names0))
        theta_idx === nothing &&
            throw(ArgumentError("could not find logit_theta or theta in SSLASSO samples."))
        spike_idx = [findfirst(==("log_tau2_spike[$j]"), names0) for j in 1:p]
        slab_idx = [findfirst(==("log_tau2_slab[$j]"), names0) for j in 1:p]
        use_log_tau = all(!isnothing, spike_idx) && all(!isnothing, slab_idx)
        tau_spike_idx = [findfirst(==("tau2_spike[$j]"), names0) for j in 1:p]
        tau_slab_idx = [findfirst(==("tau2_slab[$j]"), names0) for j in 1:p]
        use_tau = all(!isnothing, tau_spike_idx) && all(!isnothing, tau_slab_idx)
        (use_log_tau || use_tau) ||
            throw(ArgumentError("could not find SSLASSO tau2 state variables for inclusion probabilities."))
        out = zeros(size(draws, 1), p)
        for r in axes(draws, 1)
            theta = names0[theta_idx] == "theta" ? draws[r, theta_idx] : _logistic(draws[r, theta_idx])
            for j in 1:p
                b = draws[r, j]
                log_tau2_spike =
                    use_log_tau ? draws[r, Int(spike_idx[j])] : log(draws[r, Int(tau_spike_idx[j])])
                log_tau2_slab =
                    use_log_tau ? draws[r, Int(slab_idx[j])] : log(draws[r, Int(tau_slab_idx[j])])
                log_spike, log_slab, _ = _ssl_component_logpdf(b, theta, log_tau2_spike, log_tau2_slab)
                out[r, j] = exp(log_slab - _logsumexp2(log_spike, log_slab))
            end
        end
        return vec(mean(out; dims=1))
    end

    chain_names = string.(Base.invokelatest(names, result.samples))
    gamma_idx = [findfirst(==(name), chain_names) for name in gamma_names]
    if all(!isnothing, gamma_idx)
        gamma_draws = _chain_variable_draws(result.samples, Symbol.(gamma_names))
        return vec(mean(gamma_draws; dims = 1))
    end
    theta_name =
        "logit_theta" in chain_names ? :logit_theta :
        "theta" in chain_names ? :theta :
        throw(ArgumentError("could not find logit_theta or theta in SSLASSO samples."))
    beta_draws = posterior_beta_draws(result)
    theta_draws = vec(_chain_variable_draws(result.samples, [theta_name])[:, 1])
    spike_names = Symbol.(["tau2_spike[$j]" for j in 1:p])
    slab_names = Symbol.(["tau2_slab[$j]" for j in 1:p])
    log_spike_names = Symbol.(["log_tau2_spike[$j]" for j in 1:p])
    log_slab_names = Symbol.(["log_tau2_slab[$j]" for j in 1:p])
    has_tau = all(name -> string(name) in chain_names, spike_names) &&
        all(name -> string(name) in chain_names, slab_names)
    has_log_tau = all(name -> string(name) in chain_names, log_spike_names) &&
        all(name -> string(name) in chain_names, log_slab_names)
    (has_tau || has_log_tau) ||
        throw(ArgumentError("could not find SSLASSO tau2 state variables for inclusion probabilities."))
    tau_spike_draws = has_tau ? _chain_variable_draws(result.samples, spike_names) : nothing
    tau_slab_draws = has_tau ? _chain_variable_draws(result.samples, slab_names) : nothing
    log_tau_spike_draws = has_log_tau ? _chain_variable_draws(result.samples, log_spike_names) : nothing
    log_tau_slab_draws = has_log_tau ? _chain_variable_draws(result.samples, log_slab_names) : nothing
    out = zeros(size(beta_draws, 1), p)
    for r in axes(beta_draws, 1)
        theta = theta_name == :theta ? theta_draws[r] : _logistic(theta_draws[r])
        for j in 1:p
            b = beta_draws[r, j]
            log_tau2_spike = has_tau ? log(tau_spike_draws[r, j]) : log_tau_spike_draws[r, j]
            log_tau2_slab = has_tau ? log(tau_slab_draws[r, j]) : log_tau_slab_draws[r, j]
            log_spike, log_slab, _ = _ssl_component_logpdf(b, theta, log_tau2_spike, log_tau2_slab)
            out[r, j] = exp(log_slab - _logsumexp2(log_spike, log_slab))
        end
    end
    return vec(mean(out; dims=1))
end

function concordance_index(
    time::AbstractVector{<:Real},
    event::AbstractVector,
    risk_score::AbstractVector{<:Real},
)
    n = length(time)
    length(event) == n || throw(DimensionMismatch("length(event) must match length(time)."))
    length(risk_score) == n || throw(DimensionMismatch("length(risk_score) must match length(time)."))
    e = [Int(v) for v in event]
    concordant = 0.0
    comparable = 0.0
    @inbounds for i in 1:n
        if e[i] != 0
            for j in 1:n
                if time[i] < time[j]
                    comparable += 1
                    if risk_score[i] > risk_score[j]
                        concordant += 1
                    elseif risk_score[i] == risk_score[j]
                        concordant += 0.5
                    end
                end
            end
        end
    end
    comparable > 0 || return NaN
    return concordant / comparable
end
