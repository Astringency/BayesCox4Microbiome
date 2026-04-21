"""
    sample_bayes_cox(X, time, event; prior=RegularizedHorseshoePrior(), sampler=:nuts, ...)

Fit the Bayesian Cox model for compositional microbiome covariates.

The default `sampler=:nuts` follows the PDF and uses Turing.jl's NUTS implementation.
Use `sampler=:mh` for the dependency-light adaptive random-walk Metropolis fallback.
"""
function sample_bayes_cox(
    X::AbstractMatrix{<:Real},
    time::AbstractVector{<:Real},
    event::AbstractVector;
    prior::AbstractCoxPrior = RegularizedHorseshoePrior(),
    sampler::Symbol = :nuts,
    transform::Symbol = :log,
    pseudocount::Real = 1e-6,
    closure::Bool = true,
    standardize::Bool = false,
    car_weights = nothing,
    car_distance = nothing,
    car_weight_quantile::Real = 0.75,
    kwargs...,
)
    car_weights !== nothing && car_distance !== nothing &&
        throw(ArgumentError("pass either car_weights or car_distance, not both."))
    resolved_car_weights = car_weights === nothing && car_distance !== nothing ?
                           car_weight_matrix(
                               X;
                               distance = car_distance,
                               threshold_quantile = car_weight_quantile,
                           ) :
                           car_weights
    data = prepare_cox_data(
        X,
        time,
        event;
        transform = transform,
        pseudocount = pseudocount,
        closure = closure,
        standardize = standardize,
    )
    return sample_bayes_cox(
        data;
        prior = prior,
        sampler = sampler,
        car_weights = resolved_car_weights,
        kwargs...,
    )
end

function sample_bayes_cox(
    data::CoxPreparedData;
    prior::AbstractCoxPrior = RegularizedHorseshoePrior(),
    sampler::Symbol = :nuts,
    seed = nothing,
    car_weights = nothing,
    car_distance = nothing,
    car_weight_quantile::Real = 0.75,
    kwargs...,
)
    car_weights !== nothing && car_distance !== nothing &&
        throw(ArgumentError("pass either car_weights or car_distance, not both."))
    resolved_car_weights =
        if car_weights !== nothing
            car_weights
        elseif car_distance !== nothing
            data.transform == :log && !data.standardize ||
                throw(ArgumentError(
                    "car_distance for CoxPreparedData requires log-transformed, unstandardized data. " *
                    "Pass car_weights explicitly or call sample_bayes_cox with the original X matrix.",
                ))
            car_weight_matrix(
                exp.(data.z);
                distance = car_distance,
                threshold_quantile = car_weight_quantile,
            )
        else
            nothing
        end
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    if sampler == :nuts
        samples = _sample_nuts(data, prior; rng = rng, car_weights = resolved_car_weights, kwargs...)
    elseif sampler == :mh
        samples = _sample_mh(data, prior; rng = rng, car_weights = resolved_car_weights, kwargs...)
    else
        throw(ArgumentError("sampler must be :nuts or :mh."))
    end
    return CoxSamplerResult(samples, data, prior, sampler)
end

function posterior_beta_draws(result::CoxSamplerResult)
    if result.samples isa MHSamples
        p = size(result.data.z, 2)
        return result.samples.draws[:, 1:p]
    end
    throw(ArgumentError(
        "posterior_beta_draws currently supports built-in MHSamples. " *
        "For sampler=:nuts, extract beta variables directly from the returned Turing chain.",
    ))
end

function posterior_beta_summary(result::CoxSamplerResult; cred_mass::Real = 0.95)
    0 < cred_mass < 1 || throw(ArgumentError("cred_mass must be in (0, 1)."))
    draws = posterior_beta_draws(result)
    p = size(draws, 2)
    alpha = (1 - cred_mass) / 2
    beta_mean = vec(mean(draws; dims=1))
    beta_median = [median(view(draws, :, j)) for j in 1:p]
    lower = [quantile(view(draws, :, j), alpha) for j in 1:p]
    upper = [quantile(view(draws, :, j), 1 - alpha) for j in 1:p]
    return (mean = beta_mean, median = beta_median, lower = lower, upper = upper)
end

function sslasso_inclusion_probabilities(result::CoxSamplerResult)
    result.prior isa SpikeSlabLassoPrior ||
        throw(ArgumentError("sslasso_inclusion_probabilities requires SpikeSlabLassoPrior."))
    result.samples isa MHSamples ||
        throw(ArgumentError("sslasso_inclusion_probabilities currently supports built-in MHSamples."))

    prior = result.prior
    draws = result.samples.draws
    p = size(result.data.z, 2)
    out = zeros(size(draws, 1), p)
    for r in axes(draws, 1)
        theta = _logistic(draws[r, p + 1])
        for j in 1:p
            b = draws[r, j]
            log_spike = log1p(-theta) + _laplace_logpdf(b, prior.spike_scale)
            log_slab = log(theta) + _laplace_logpdf(b, prior.slab_scale)
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
