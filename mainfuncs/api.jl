"""
    sample_bayes_cox(X, time, event; prior=RegularizedHorseshoePrior(), sampler=:nuts, ...)

Fit the Bayesian Cox model for compositional microbiome covariates.

The default `sampler=:nuts` follows the PDF and uses Turing.jl's NUTS implementation.
Use `sampler=:mh` for the dependency-light adaptive random-walk Metropolis fallback.
"""
function _cox_ridge_gradient(data::CoxPreparedData, beta::AbstractVector)
    z = data.z
    eta = z * beta
    g = zeros(Float64, size(z, 2))
    @inbounds for i in eachindex(data.event)
        data.event[i] == 0 && continue
        denom = 0.0
        weighted = zeros(Float64, size(z, 2))
        ti = data.time[i]
        for r in eachindex(data.time)
            if data.time[r] >= ti
                w = exp(clamp(eta[r], -30.0, 30.0))
                denom += w
                @views weighted .+= w .* z[r, :]
            end
        end
        @views g .+= z[i, :] .- weighted ./ denom
    end
    return g
end

function _estimate_cox_pseudo_mu(data::CoxPreparedData; maxiter::Integer = 50, ridge::Real = 1.0)
    p = size(data.z, 2)
    beta = zeros(Float64, p)
    objective(b) = cox_logpartiallikelihood(data, b) - 0.5 * ridge * sum(abs2, b)
    current = objective(beta)
    for _ in 1:maxiter
        grad = _cox_ridge_gradient(data, beta) .- ridge .* beta
        grad .-= mean(grad)
        norm(grad) < 1e-6 && break
        step = 0.05 / max(1.0, norm(grad) / sqrt(p))
        accepted = false
        for _ in 1:12
            proposal = beta .+ step .* grad
            proposal .-= mean(proposal)
            prop_obj = objective(proposal)
            if isfinite(prop_obj) && prop_obj >= current
                beta = proposal
                current = prop_obj
                accepted = true
                break
            end
            step /= 2
        end
        accepted || break
    end
    eta = clamp.(data.z * beta, -30.0, 30.0)
    return mean(exp.(eta))
end

function _p0_prior_name(name)
    key = lowercase(replace(string(name), "_" => "-", " " => "-"))
    return key in (
        "hs",
        "horseshoe",
        "rhs",
        "regularized-horseshoe",
        "new-rhs",
        "newrhs",
        "est-rhs",
        "estrhs",
        "cox-hs",
        "coxhs",
        "cox-rhs",
        "coxrhs",
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

Construct one of the eight prior specifications used in the simulation section
of `docs/cox4microbiome_detail.pdf`: `hs`, `rhs`, `new-rhs`, `est-rhs`,
`cox-hs`, `cox-rhs`, `lasso`, and `sslasso`.

When `p0` is `nothing`, HS/RHS-style priors estimate `p0` by univariate Cox
screening and use the number of covariates significant at `p0_alpha`.
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
    cox_estimate_maxiter::Integer = 50,
    p0_alpha::Real = 0.05,
    p0_min::Integer = 1,
    p0_max = nothing,
)
    key = lowercase(replace(string(name), "_" => "-", " " => "-"))
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
    if key in ("hs", "horseshoe")
        return HorseshoePrior(tau_scale = 1.0, use_cox_tau0 = false, p0 = p0_value)
    elseif key in ("rhs", "regularized-horseshoe")
        return RegularizedHorseshoePrior(
            use_cox_tau0 = false,
            p0 = p0_value,
            tau_formula = :linear,
            pseudo_sigma = 10.0,
            sample_pseudo_sigma = true,
            pseudo_sigma_df = 3.0,
            pseudo_sigma_scale = 10.0,
            slab_df = 4.0,
            slab_s2 = 2.0,
        )
    elseif key in ("new-rhs", "newrhs")
        return RegularizedHorseshoePrior(
            use_cox_tau0 = false,
            p0 = p0_value,
            tau_formula = :linear,
            pseudo_sigma = 1.0,
            slab_df = 4.0,
            slab_s2 = 2.0,
        )
    elseif key in ("est-rhs", "estrhs")
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
        muhat = _estimate_cox_pseudo_mu(data; maxiter = cox_estimate_maxiter)
        return RegularizedHorseshoePrior(
            use_cox_tau0 = false,
            p0 = p0_value,
            tau_formula = :linear,
            pseudo_sigma = sqrt(inv(max(muhat, eps(Float64)))),
            slab_df = 4.0,
            slab_s2 = 2.0,
        )
    elseif key in ("cox-hs", "coxhs")
        return HorseshoePrior(use_cox_tau0 = true, p0 = p0_value)
    elseif key in ("cox-rhs", "coxrhs")
        return RegularizedHorseshoePrior(
            use_cox_tau0 = true,
            p0 = p0_value,
            tau_formula = :cox,
            slab_df = 4.0,
            slab_s2 = 2.0,
        )
    elseif key in ("lasso", "bayesian-lasso")
        return BayesianLassoPrior(lambda = nothing, lambda2_shape = 1.0, lambda2_rate = 0.1)
    elseif key in ("sslasso", "spike-slab", "spike-and-slab")
        return realdata ? SpikeSlabLassoPrior(spike_scale = 1.0, slab_scale = 10.0) :
                          SpikeSlabLassoPrior(spike_scale = 0.04, slab_scale = 0.5)
    end
    throw(ArgumentError("unsupported paper prior name: $name"))
end

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
    use_car::Bool = false,
    car_weights = nothing,
    car_distance = nothing,
    car_weight_quantile::Real = 0.75,
    kwargs...,
)
    car_weights !== nothing && car_distance !== nothing && use_car &&
        throw(ArgumentError("pass either car_weights or car_distance, not both."))
    resolved_car_weights =
        if !use_car
            nothing
        elseif car_weights !== nothing
            car_weights
        else
            car_weight_matrix(
                X;
                distance = car_distance === nothing ? :bray : car_distance,
                threshold_quantile = car_weight_quantile,
            )
        end
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
        use_car = use_car,
        car_weights = resolved_car_weights,
        kwargs...,
    )
end

function sample_bayes_cox(
    data::CoxPreparedData;
    prior::AbstractCoxPrior = RegularizedHorseshoePrior(),
    sampler::Symbol = :nuts,
    seed = nothing,
    use_car::Bool = false,
    car_weights = nothing,
    car_distance = nothing,
    car_weight_quantile::Real = 0.75,
    kwargs...,
)
    car_weights !== nothing && car_distance !== nothing && use_car &&
        throw(ArgumentError("pass either car_weights or car_distance, not both."))
    resolved_car_weights =
        if !use_car
            nothing
        elseif car_weights !== nothing
            car_weights
        else
            data.transform == :log && !data.standardize ||
                throw(ArgumentError(
                    "car_distance for CoxPreparedData requires log-transformed, unstandardized data. " *
                    "Pass car_weights explicitly or call sample_bayes_cox with the original X matrix.",
                ))
            car_weight_matrix(
                exp.(data.z);
                distance = car_distance === nothing ? :bray : car_distance,
                threshold_quantile = car_weight_quantile,
            )
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
