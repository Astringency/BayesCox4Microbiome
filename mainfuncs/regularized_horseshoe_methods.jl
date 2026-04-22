function _validate_prior(prior::RegularizedHorseshoePrior)
    _check_positive(prior.tau_scale, "tau_scale")
    _check_positive(prior.p0, "p0")
    _check_positive(prior.local_scale, "local_scale")
    _check_positive(prior.slab_df, "slab_df")
    _check_positive(prior.slab_s2, "slab_s2")
    prior.tau_formula in (:auto, :fixed, :linear, :cox) ||
        throw(ArgumentError("tau_formula must be :auto, :fixed, :linear, or :cox."))
    if prior.pseudo_sigma !== nothing
        _check_positive(prior.pseudo_sigma, "pseudo_sigma")
    end
    _check_positive(prior.pseudo_sigma_df, "pseudo_sigma_df")
    _check_positive(prior.pseudo_sigma_scale, "pseudo_sigma_scale")
    return prior
end

function _resolved_tau_scale(
    prior::RegularizedHorseshoePrior,
    p::Integer,
    n_events::Integer;
    n_obs::Union{Nothing,Integer} = nothing,
)
    formula = prior.tau_formula == :auto ? (prior.use_cox_tau0 ? :cox : :fixed) : prior.tau_formula
    default_sigma = prior.sample_pseudo_sigma ? sqrt(prior.pseudo_sigma_scale) : 1.0
    n_linear = n_obs === nothing ? n_events : n_obs
    return formula == :cox ? cox_tau0(p, prior.p0, n_events) :
           formula == :linear ? linear_tau0(p, prior.p0, n_linear, something(prior.pseudo_sigma, default_sigma)) :
           prior.tau_scale
end

function _resolved_tau_scale(
    prior::RegularizedHorseshoePrior,
    data::CoxPreparedData;
    pseudo_sigma = prior.pseudo_sigma,
)
    p = size(data.z, 2)
    formula = prior.tau_formula == :auto ? (prior.use_cox_tau0 ? :cox : :fixed) : prior.tau_formula
    return formula == :cox ? cox_tau0(p, prior.p0, count(!=(0), data.event)) :
           formula == :linear ? linear_tau0(p, prior.p0, size(data.z, 1), something(pseudo_sigma, 1.0)) :
           prior.tau_scale
end

function _state_names(prior::RegularizedHorseshoePrior, p::Integer)
    names = vcat(
        ["beta[$j]" for j in 1:p],
        ["log_tau"],
        ["log_lambda[$j]" for j in 1:p],
        ["log_c2"],
    )
    if prior.sample_pseudo_sigma
        push!(names, "log_pseudo_sigma2")
    end
    return names
end

function initial_state(
    prior::RegularizedHorseshoePrior,
    p::Integer,
    n_events::Integer;
    n_obs::Union{Nothing,Integer} = nothing,
    beta0 = zeros(Float64, p),
)
    _validate_prior(prior)
    length(beta0) == p || throw(DimensionMismatch("length(beta0) must be p."))
    tau0 = _resolved_tau_scale(prior, p, n_events; n_obs = n_obs)
    init = vcat(collect(Float64, beta0), [log(tau0)], zeros(p), [log(prior.slab_s2)])
    if prior.sample_pseudo_sigma
        pseudo_sigma2_init = prior.pseudo_sigma === nothing ? prior.pseudo_sigma_scale : prior.pseudo_sigma^2
        push!(init, log(pseudo_sigma2_init))
    end
    return init
end

function _prior_logposterior(
    u::AbstractVector,
    data::CoxPreparedData,
    prior::RegularizedHorseshoePrior;
    car_weights = nothing,
    car_precision::Real = 1.0,
)
    p = size(data.z, 2)
    expected = prior.sample_pseudo_sigma ? 2p + 3 : 2p + 2
    length(u) == expected ||
        throw(DimensionMismatch("wrong state length for RegularizedHorseshoePrior."))
    beta = view(u, 1:p)
    log_tau = u[p + 1]
    log_lambda = view(u, (p + 2):(2p + 1))
    log_c2 = u[2p + 2]
    log_pseudo_sigma2 = prior.sample_pseudo_sigma ? u[2p + 3] : nothing
    tau = exp(log_tau)
    lambda = exp.(log_lambda)
    c2 = exp(log_c2)
    pseudo_sigma2 = prior.sample_pseudo_sigma ? exp(clamp(log_pseudo_sigma2, -20, 20)) : nothing
    pseudo_sigma = prior.sample_pseudo_sigma ? sqrt(pseudo_sigma2) : prior.pseudo_sigma
    tau_scale = _resolved_tau_scale(prior, data; pseudo_sigma = pseudo_sigma)

    lp = _halfcauchy_logpdf(tau, tau_scale) + log_tau
    if prior.sample_pseudo_sigma
        lp += _log_positive_t_pdf(
            pseudo_sigma2,
            prior.pseudo_sigma_df,
            prior.pseudo_sigma_scale,
        ) + log_pseudo_sigma2
    end
    for j in 1:p
        lp += _halfcauchy_logpdf(lambda[j], prior.local_scale) + log_lambda[j]
        log_beta_scale = _regularized_horseshoe_beta_logscale(tau, lambda[j], c2)
        lp += _normal_zero_logpdf_from_logscale(beta[j], log_beta_scale)
    end
    lp += logpdf(InverseGamma(prior.slab_df / 2, prior.slab_df * prior.slab_s2 / 2), c2) + log_c2
    lp += _car_logprior(log_lambda, car_weights, car_precision)
    return lp
end
