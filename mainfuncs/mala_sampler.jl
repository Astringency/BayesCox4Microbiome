function _mala_logq(y::AbstractVector, x::AbstractVector, grad_x::AbstractVector, step_size::Real)
    gamma_L = Float64(step_size)
    diff = y .- x .- gamma_L .* grad_x
    return -0.25 / gamma_L * sum(abs2, diff)
end

function _mala_step(
    logpost::Function,
    gradient::Function,
    u::Vector{Float64},
    current_lp::Float64,
    gamma_L::Float64,
    rng::AbstractRNG,
)
    grad_u = gradient(u)
    proposal = u .+ gamma_L .* grad_u .+ sqrt(2 * gamma_L) .* randn(rng, length(u))
    prop_lp = logpost(proposal)
    if !isfinite(prop_lp)
        return u, current_lp, false
    end
    grad_prop = gradient(proposal)
    log_alpha = prop_lp + _mala_logq(u, proposal, grad_prop, gamma_L) -
        current_lp - _mala_logq(proposal, u, grad_u, gamma_L)
    accepted = log(rand(rng)) < log_alpha
    return accepted ? proposal : u, accepted ? prop_lp : current_lp, accepted
end

function _full_logposterior_gradient_beta(
    u::AbstractVector,
    data::CoxPreparedData,
    prior::AbstractCoxPrior;
    constraint::Symbol,
    soft_center_scale::Real,
    prox_sum0_lambda::Real,
)
    p = size(data.z, 2)
    beta = view(u, 1:p)
    grad = cox_score_gradient(data, beta)
    _constraint_gradient_penalty!(
        grad,
        beta;
        constraint = constraint,
        soft_center_scale = soft_center_scale,
        prox_sum0_lambda = prox_sum0_lambda,
    )

    if prior isa HorseshoePrior
        log_tau = u[p + 1]
        log_lambda = view(u, (p + 2):(2p + 1))
        @inbounds for j in 1:p
            scale2 = exp(2 * (log_tau + log_lambda[j]))
            grad[j] -= beta[j] / scale2
        end
    elseif prior isa RegularizedHorseshoePrior
        log_tau = u[p + 1]
        log_lambda = view(u, (p + 2):(2p + 1))
        log_c2 = u[2p + 2]
        c2 = exp(log_c2)
        @inbounds for j in 1:p
            log_scale = _regularized_horseshoe_beta_logscale(exp(log_tau), exp(log_lambda[j]), c2)
            grad[j] -= beta[j] * exp(-2 * log_scale)
        end
    elseif prior isa BayesianLassoPrior
        log_tau2 = view(u, (p + 1):(2p))
        @inbounds for j in 1:p
            grad[j] -= beta[j] / exp(log_tau2[j])
        end
    elseif prior isa SpikeSlabLassoPrior
        theta = _logistic(u[p + 1])
        theta = clamp(theta, eps(Float64), 1 - eps(Float64))
        @inbounds for j in 1:p
            b = beta[j]
            sgn = b > 0 ? 1.0 : b < 0 ? -1.0 : 0.0
            log_spike = log1p(-theta) + _laplace_logpdf(b, prior.spike_scale)
            log_slab = log(theta) + _laplace_logpdf(b, prior.slab_scale)
            norm = _logsumexp2(log_spike, log_slab)
            prob_spike = exp(log_spike - norm)
            prob_slab = exp(log_slab - norm)
            grad[j] -= sgn * (prob_spike / prior.spike_scale + prob_slab / prior.slab_scale)
        end
    else
        throw(ArgumentError("unsupported prior type for MALA: $(typeof(prior))"))
    end
    return grad
end

function _hybrid_beta_mala(
    data::CoxPreparedData,
    prior::AbstractCoxPrior;
    n_samples::Integer,
    warmup::Integer,
    thin::Integer,
    rng::AbstractRNG,
    initial,
    mala_step_size::Real,
    mala_target_accept::Real,
    mala_adapt_every::Integer,
    mala_min_step::Real,
    mala_max_step::Real,
    initial_step::Union{Real,AbstractVector{<:Real}},
    target_accept::Real,
    adapt_every::Integer,
    min_step::Real,
    max_step::Real,
    constraint::Symbol,
    soft_center_scale::Real,
    prox_sum0_lambda::Real,
    car_weights,
    use_car::Bool,
    car_precision::Real,
    progress::Bool,
)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive."))
    warmup >= 0 || throw(ArgumentError("warmup must be nonnegative."))
    thin > 0 || throw(ArgumentError("thin must be positive."))
    mala_adapt_every > 0 || throw(ArgumentError("mala_adapt_every must be positive."))
    gamma_L = _check_positive(mala_step_size, "mala_step_size")
    gamma_min = _check_positive(mala_min_step, "mala_min_step")
    gamma_max = _check_positive(mala_max_step, "mala_max_step")
    0 < mala_target_accept < 1 || throw(ArgumentError("mala_target_accept must be in (0, 1)."))

    p = size(data.z, 2)
    init = collect(Float64, initial)
    names = _state_names(prior, p)
    d = length(init)
    length(names) == d || throw(DimensionMismatch("initial state length does not match prior state."))
    hdim = d - p
    hyper_step =
        if initial_step isa Real
            fill(Float64(initial_step), hdim)
        else
            raw_step = collect(Float64, initial_step)
            length(raw_step) == hdim ? raw_step :
            length(raw_step) == d ? raw_step[(p + 1):d] :
            throw(DimensionMismatch("initial_step must have length $(hdim) for hyperparameters or full state length $(d)."))
        end
    length(hyper_step) == hdim || throw(DimensionMismatch("initial_step hyperparameter length mismatch."))
    all(>(0), hyper_step) || throw(ArgumentError("initial_step entries must be positive."))

    logpost = u -> logposterior(
        u,
        data,
        prior;
        constraint = constraint,
        soft_center_scale = soft_center_scale,
        prox_sum0_lambda = prox_sum0_lambda,
        car_weights = car_weights,
        use_car = use_car,
        car_precision = car_precision,
    )
    beta_grad = u -> _full_logposterior_gradient_beta(
        u,
        data,
        prior;
        constraint = constraint,
        soft_center_scale = soft_center_scale,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    beta_logpost = b -> begin
        uu = copy(init)
        uu[1:p] .= b
        return logpost(uu)
    end
    beta_gradient = b -> begin
        uu = copy(init)
        uu[1:p] .= b
        return beta_grad(uu)
    end

    total_iter = warmup + n_samples * thin
    draws = Matrix{Float64}(undef, n_samples, d)
    lps = Vector{Float64}(undef, n_samples)
    u = init
    current_lp = logpost(u)
    isfinite(current_lp) || throw(ArgumentError("initial state has non-finite log posterior."))

    accept_warmup = 0
    accept_sample = 0
    accept_window = 0
    hyper_accept_window = 0
    hyper_accept_warmup = 0
    hyper_accept_sample = 0
    saved = 0
    prog = progress ? Progress(total_iter; desc = "MALA sampling", showspeed = true, dt = 0.0) : nothing

    for iter in 1:total_iter
        beta_logpost_current = b -> begin
            uu = copy(u)
            uu[1:p] .= b
            return logpost(uu)
        end
        beta_gradient_current = b -> begin
            uu = copy(u)
            uu[1:p] .= b
            return beta_grad(uu)
        end
        beta_new, beta_lp, accepted = _mala_step(
            beta_logpost_current,
            beta_gradient_current,
            collect(view(u, 1:p)),
            current_lp,
            gamma_L,
            rng,
        )
        if accepted
            u[1:p] .= beta_new
            current_lp = beta_lp
            if iter <= warmup
                accept_warmup += 1
                accept_window += 1
            else
                accept_sample += 1
            end
        end

        if hdim > 0
            prop = copy(u)
            prop[(p + 1):d] .+= hyper_step .* randn(rng, hdim)
            prop_lp = logpost(prop)
            hyper_accepted = isfinite(prop_lp) && log(rand(rng)) < prop_lp - current_lp
            if hyper_accepted
                u = prop
                current_lp = prop_lp
                if iter <= warmup
                    hyper_accept_warmup += 1
                    hyper_accept_window += 1
                else
                    hyper_accept_sample += 1
                end
            end
        end

        if iter <= warmup && iter % mala_adapt_every == 0
            window_rate = accept_window / mala_adapt_every
            gamma_L = clamp(gamma_L * exp(clamp(window_rate - mala_target_accept, -0.5, 0.5)), gamma_min, gamma_max)
            accept_window = 0
        end
        if hdim > 0 && iter <= warmup && iter % adapt_every == 0
            window_rate = hyper_accept_window / adapt_every
            hyper_step .= clamp.(hyper_step .* exp(clamp(window_rate - target_accept, -0.25, 0.25)), min_step, max_step)
            hyper_accept_window = 0
        end

        if iter > warmup && (iter - warmup) % thin == 0
            saved += 1
            @views draws[saved, :] .= u
            lps[saved] = current_lp
        end
        phase = iter <= warmup ? "warmup $(iter)/$(warmup)" : "sample $(saved)/$(n_samples)"
        prog !== nothing && next!(prog; showvalues = [(:phase, phase), (:saved, saved)])
    end
    prog !== nothing && finish!(prog)

    sample_iters = n_samples * thin
    warmup_rate = warmup == 0 ? NaN : accept_warmup / warmup
    sample_rate = sample_iters == 0 ? NaN : accept_sample / sample_iters
    final_step = vcat(fill(gamma_L, p), hyper_step)
    return MHSamples(draws, lps, names, sample_rate, warmup_rate, final_step)
end

function _lasso_prox_logposterior(
    u::AbstractVector,
    data::CoxPreparedData,
    prior::BayesianLassoPrior;
    constraint::Symbol,
    soft_center_scale::Real,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_rate::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = size(data.z, 2)
    beta = view(u, 1:p)
    eta = u[p + 1]
    alpha = exp(clamp(eta, -700.0, 700.0))
    lp = cox_logpartiallikelihood(data, beta)
    lp += _constraint_logpenalty(
        beta;
        constraint = constraint,
        soft_center_scale = soft_center_scale,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    lp -= l1_moreau(beta, alpha, prox_l1_lambda; prox_root_tol = prox_root_tol, prox_root_maxiter = prox_root_maxiter)
    lp += -Float64(alpha_prior_rate) * exp(-eta) - Float64(alpha_prior_shape) * eta
    return lp
end

function _lasso_prox_gradient(
    u::AbstractVector,
    data::CoxPreparedData;
    constraint::Symbol,
    soft_center_scale::Real,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_rate::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = size(data.z, 2)
    beta = view(u, 1:p)
    eta = u[p + 1]
    alpha = exp(clamp(eta, -700.0, 700.0))
    lambda_l1 = _check_positive(prox_l1_lambda, "prox_l1_lambda")
    prox = project_l1_epigraph(beta, alpha; prox_root_tol = prox_root_tol, prox_root_maxiter = prox_root_maxiter)
    grad = zeros(Float64, p + 1)
    grad[1:p] .= cox_score_gradient(data, beta)
    _constraint_gradient_penalty!(
        view(grad, 1:p),
        beta;
        constraint = constraint,
        soft_center_scale = soft_center_scale,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    @inbounds for j in 1:p
        grad[j] -= (Float64(beta[j]) - prox.beta[j]) / lambda_l1
    end
    grad[p + 1] = Float64(alpha_prior_rate) * exp(-eta) - Float64(alpha_prior_shape) -
        alpha * (alpha - prox.alpha) / lambda_l1
    return grad
end

function _sample_lasso_prox_mala(
    data::CoxPreparedData,
    prior::BayesianLassoPrior;
    n_samples::Integer,
    warmup::Integer,
    thin::Integer,
    rng::AbstractRNG,
    initial,
    mala_step_size::Real,
    mala_target_accept::Real,
    mala_adapt_every::Integer,
    mala_min_step::Real,
    mala_max_step::Real,
    constraint::Symbol,
    soft_center_scale::Real,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_rate::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
    progress::Bool,
)
    p = size(data.z, 2)
    init = initial === nothing ? vcat(zeros(Float64, p), [0.0]) : collect(Float64, initial)
    length(init) == p + 1 || throw(DimensionMismatch("initial length must be p + 1 for prox LASSO MALA."))
    names = vcat(["beta[$j]" for j in 1:p], ["eta"])
    logpost = u -> _lasso_prox_logposterior(
        u,
        data,
        prior;
        constraint = constraint,
        soft_center_scale = soft_center_scale,
        prox_sum0_lambda = prox_sum0_lambda,
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_rate = alpha_prior_rate,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    gradient = u -> _lasso_prox_gradient(
        u,
        data;
        constraint = constraint,
        soft_center_scale = soft_center_scale,
        prox_sum0_lambda = prox_sum0_lambda,
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_rate = alpha_prior_rate,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    return _run_full_mala(logpost, gradient, init, names; n_samples = n_samples, warmup = warmup, thin = thin, rng = rng,
        mala_step_size = mala_step_size, mala_target_accept = mala_target_accept, mala_adapt_every = mala_adapt_every,
        mala_min_step = mala_min_step, mala_max_step = mala_max_step, progress = progress)
end

function _run_full_mala(
    logpost::Function,
    gradient::Function,
    init::AbstractVector,
    names::Vector{String};
    n_samples::Integer,
    warmup::Integer,
    thin::Integer,
    rng::AbstractRNG,
    mala_step_size::Real,
    mala_target_accept::Real,
    mala_adapt_every::Integer,
    mala_min_step::Real,
    mala_max_step::Real,
    progress::Bool,
)
    gamma_L = _check_positive(mala_step_size, "mala_step_size")
    gamma_min = _check_positive(mala_min_step, "mala_min_step")
    gamma_max = _check_positive(mala_max_step, "mala_max_step")
    total_iter = warmup + n_samples * thin
    u = collect(Float64, init)
    d = length(u)
    draws = Matrix{Float64}(undef, n_samples, d)
    lps = Vector{Float64}(undef, n_samples)
    current_lp = logpost(u)
    isfinite(current_lp) || throw(ArgumentError("initial state has non-finite log posterior."))
    accept_warmup = 0
    accept_sample = 0
    accept_window = 0
    saved = 0
    prog = progress ? Progress(total_iter; desc = "MALA sampling", showspeed = true, dt = 0.0) : nothing
    for iter in 1:total_iter
        u_new, lp_new, accepted = _mala_step(logpost, gradient, u, current_lp, gamma_L, rng)
        if accepted
            u = u_new
            current_lp = lp_new
            if iter <= warmup
                accept_warmup += 1
                accept_window += 1
            else
                accept_sample += 1
            end
        end
        if iter <= warmup && iter % mala_adapt_every == 0
            window_rate = accept_window / mala_adapt_every
            gamma_L = clamp(gamma_L * exp(clamp(window_rate - mala_target_accept, -0.5, 0.5)), gamma_min, gamma_max)
            accept_window = 0
        elseif iter > warmup && (iter - warmup) % thin == 0
            saved += 1
            @views draws[saved, :] .= u
            lps[saved] = current_lp
        end
        phase = iter <= warmup ? "warmup $(iter)/$(warmup)" : "sample $(saved)/$(n_samples)"
        prog !== nothing && next!(prog; showvalues = [(:phase, phase), (:saved, saved)])
    end
    prog !== nothing && finish!(prog)
    sample_iters = n_samples * thin
    return MHSamples(
        draws,
        lps,
        names,
        sample_iters == 0 ? NaN : accept_sample / sample_iters,
        warmup == 0 ? NaN : accept_warmup / warmup,
        fill(gamma_L, d),
    )
end

function _ssl_weights(prior::SpikeSlabLassoPrior, gamma::AbstractVector{Bool})
    return [inv(gamma[j] ? prior.slab_scale : prior.spike_scale) for j in eachindex(gamma)]
end

function _ssl_prox_logposterior(
    u::AbstractVector,
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior,
    gamma::AbstractVector{Bool},
    theta::Real;
    constraint::Symbol,
    soft_center_scale::Real,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_rate::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = size(data.z, 2)
    beta = view(u, 1:p)
    eta = u[p + 1]
    alpha = exp(clamp(eta, -700.0, 700.0))
    weights = _ssl_weights(prior, gamma)
    lp = cox_logpartiallikelihood(data, beta)
    lp += _constraint_logpenalty(
        beta;
        constraint = constraint,
        soft_center_scale = soft_center_scale,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    lp -= weighted_l1_moreau(beta, alpha, weights, prox_l1_lambda; prox_root_tol = prox_root_tol, prox_root_maxiter = prox_root_maxiter)
    lp += -Float64(alpha_prior_rate) * exp(-eta) - Float64(alpha_prior_shape) * eta
    theta_clamped = clamp(theta, eps(Float64), 1 - eps(Float64))
    lp += sum(gamma) * log(theta_clamped) + (length(gamma) - sum(gamma)) * log1p(-theta_clamped)
    return lp
end

function _ssl_prox_gradient(
    u::AbstractVector,
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior,
    gamma::AbstractVector{Bool};
    constraint::Symbol,
    soft_center_scale::Real,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_rate::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = size(data.z, 2)
    beta = view(u, 1:p)
    eta = u[p + 1]
    alpha = exp(clamp(eta, -700.0, 700.0))
    weights = _ssl_weights(prior, gamma)
    lambda_l1 = _check_positive(prox_l1_lambda, "prox_l1_lambda")
    prox = project_weighted_l1_epigraph(beta, alpha, weights; prox_root_tol = prox_root_tol, prox_root_maxiter = prox_root_maxiter)
    grad = zeros(Float64, p + 1)
    grad[1:p] .= cox_score_gradient(data, beta)
    _constraint_gradient_penalty!(
        view(grad, 1:p),
        beta;
        constraint = constraint,
        soft_center_scale = soft_center_scale,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    @inbounds for j in 1:p
        grad[j] -= (Float64(beta[j]) - prox.beta[j]) / lambda_l1
    end
    grad[p + 1] = Float64(alpha_prior_rate) * exp(-eta) - Float64(alpha_prior_shape) -
        alpha * (alpha - prox.alpha) / lambda_l1
    return grad
end

function _sample_ssl_prox_mala(
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior;
    n_samples::Integer,
    warmup::Integer,
    thin::Integer,
    rng::AbstractRNG,
    initial,
    mala_step_size::Real,
    mala_target_accept::Real,
    mala_adapt_every::Integer,
    mala_min_step::Real,
    mala_max_step::Real,
    constraint::Symbol,
    soft_center_scale::Real,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_rate::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
    progress::Bool,
)
    # Conditional SSLASSO proximal epigraph sampler: gamma is discrete and is
    # updated by Bernoulli Gibbs, while MALA only targets continuous beta/eta.
    # This is distinct from the marginal mixture-Laplace SSLASSO target.
    p = size(data.z, 2)
    u = initial === nothing ? vcat(zeros(Float64, p), [0.0]) : collect(Float64, initial)
    length(u) == p + 1 || throw(DimensionMismatch("initial length must be p + 1 for prox SSLASSO MALA."))
    gamma = trues(p)
    theta = 0.5
    names = vcat(["beta[$j]" for j in 1:p], ["eta"], ["gamma[$j]" for j in 1:p], ["theta"])
    d = length(names)
    draws = Matrix{Float64}(undef, n_samples, d)
    lps = Vector{Float64}(undef, n_samples)
    gamma_L = _check_positive(mala_step_size, "mala_step_size")
    gamma_min = _check_positive(mala_min_step, "mala_min_step")
    gamma_max = _check_positive(mala_max_step, "mala_max_step")
    total_iter = warmup + n_samples * thin
    logpost = uu -> _ssl_prox_logposterior(
        uu,
        data,
        prior,
        gamma,
        theta;
        constraint = constraint,
        soft_center_scale = soft_center_scale,
        prox_sum0_lambda = prox_sum0_lambda,
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_rate = alpha_prior_rate,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    current_lp = logpost(u)
    isfinite(current_lp) || throw(ArgumentError("initial state has non-finite log posterior."))
    accept_warmup = 0
    accept_sample = 0
    accept_window = 0
    saved = 0
    prog = progress ? Progress(total_iter; desc = "SSLASSO prox MALA", showspeed = true, dt = 0.0) : nothing
    for iter in 1:total_iter
        logpost_current = uu -> _ssl_prox_logposterior(
            uu,
            data,
            prior,
            gamma,
            theta;
            constraint = constraint,
            soft_center_scale = soft_center_scale,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = alpha_prior_shape,
            alpha_prior_rate = alpha_prior_rate,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
        )
        gradient_current = uu -> _ssl_prox_gradient(
            uu,
            data,
            prior,
            gamma;
            constraint = constraint,
            soft_center_scale = soft_center_scale,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = alpha_prior_shape,
            alpha_prior_rate = alpha_prior_rate,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
        )
        u_new, lp_new, accepted = _mala_step(logpost_current, gradient_current, u, current_lp, gamma_L, rng)
        if accepted
            u = u_new
            current_lp = lp_new
            if iter <= warmup
                accept_warmup += 1
                accept_window += 1
            else
                accept_sample += 1
            end
        end

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
        current_lp = logpost_current(u)

        if iter <= warmup && iter % mala_adapt_every == 0
            window_rate = accept_window / mala_adapt_every
            gamma_L = clamp(gamma_L * exp(clamp(window_rate - mala_target_accept, -0.5, 0.5)), gamma_min, gamma_max)
            accept_window = 0
        elseif iter > warmup && (iter - warmup) % thin == 0
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
                soft_center_scale = soft_center_scale,
                prox_sum0_lambda = prox_sum0_lambda,
                prox_l1_lambda = prox_l1_lambda,
                alpha_prior_shape = alpha_prior_shape,
                alpha_prior_rate = alpha_prior_rate,
                prox_root_tol = prox_root_tol,
                prox_root_maxiter = prox_root_maxiter,
            )
        end
        phase = iter <= warmup ? "warmup $(iter)/$(warmup)" : "sample $(saved)/$(n_samples)"
        prog !== nothing && next!(prog; showvalues = [(:phase, phase), (:saved, saved)])
    end
    prog !== nothing && finish!(prog)
    sample_iters = n_samples * thin
    return MHSamples(draws, lps, names, sample_iters == 0 ? NaN : accept_sample / sample_iters,
        warmup == 0 ? NaN : accept_warmup / warmup, fill(gamma_L, d))
end

function _sample_mala(
    data::CoxPreparedData,
    prior::AbstractCoxPrior;
    n_samples::Integer = 5000,
    warmup::Integer = 2000,
    thin::Integer = 1,
    rng::AbstractRNG = Random.default_rng(),
    initial = nothing,
    initial_step::Union{Real,AbstractVector{<:Real}} = 0.05,
    target_accept::Real = 0.234,
    adapt_every::Integer = 50,
    min_step::Real = 1e-5,
    max_step::Real = 1.0,
    mala_step_size::Real = 1e-4,
    mala_target_accept::Real = 0.574,
    mala_adapt_every::Integer = 50,
    mala_min_step::Real = 1e-8,
    mala_max_step::Real = 1.0,
    mala_precondition::Bool = false,
    constraint::Symbol = :soft_center,
    soft_center = nothing,
    soft_center_scale::Union{Nothing,Real} = nothing,
    prox_sum0_lambda::Real = 1e-3,
    prox_l1_lambda::Real = 1e-3,
    alpha_prior_shape = nothing,
    alpha_prior_rate = nothing,
    prox_root_tol::Real = 1e-10,
    prox_root_maxiter::Integer = 100,
    car_weights = nothing,
    use_car::Bool = car_weights !== nothing,
    car_precision::Real = 1.0,
    progress::Bool = true,
)
    mala_precondition &&
        throw(ArgumentError("mala_precondition=true is not implemented yet; the current MALA uses identity mass."))
    _validate_prior(prior)
    p = size(data.z, 2)
    n_events = count(!=(0), data.event)
    resolved_constraint = _resolve_constraint(constraint, soft_center)
    scale = soft_center_scale === nothing ? _default_soft_center_scale(p) : Float64(soft_center_scale)
    _check_positive(prox_sum0_lambda, "prox_sum0_lambda")
    _check_positive(prox_l1_lambda, "prox_l1_lambda")
    _check_positive(prox_root_tol, "prox_root_tol")
    prox_root_maxiter > 0 || throw(ArgumentError("prox_root_maxiter must be positive."))
    a_shape = Float64(something(alpha_prior_shape, prior isa BayesianLassoPrior ? prior.lambda2_shape : 1.0))
    a_rate = Float64(something(alpha_prior_rate, prior isa BayesianLassoPrior ? prior.lambda2_rate : 1.0))
    _check_positive(a_shape, "alpha_prior_shape")
    _check_positive(a_rate, "alpha_prior_rate")

    if prior isa BayesianLassoPrior && resolved_constraint == :prox
        return _sample_lasso_prox_mala(
            data,
            prior;
            n_samples = n_samples,
            warmup = warmup,
            thin = thin,
            rng = rng,
            initial = initial,
            mala_step_size = mala_step_size,
            mala_target_accept = mala_target_accept,
            mala_adapt_every = mala_adapt_every,
            mala_min_step = mala_min_step,
            mala_max_step = mala_max_step,
            constraint = resolved_constraint,
            soft_center_scale = scale,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = a_shape,
            alpha_prior_rate = a_rate,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
            progress = progress,
        )
    elseif prior isa SpikeSlabLassoPrior && resolved_constraint == :prox
        return _sample_ssl_prox_mala(
            data,
            prior;
            n_samples = n_samples,
            warmup = warmup,
            thin = thin,
            rng = rng,
            initial = initial,
            mala_step_size = mala_step_size,
            mala_target_accept = mala_target_accept,
            mala_adapt_every = mala_adapt_every,
            mala_min_step = mala_min_step,
            mala_max_step = mala_max_step,
            constraint = resolved_constraint,
            soft_center_scale = scale,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = a_shape,
            alpha_prior_rate = a_rate,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
            progress = progress,
        )
    end

    init =
        if initial !== nothing
            collect(Float64, initial)
        elseif prior isa RegularizedHorseshoePrior
            initial_state(prior, p, n_events; n_obs = size(data.z, 1))
        else
            initial_state(prior, p, n_events)
        end
    return _hybrid_beta_mala(
        data,
        prior;
        n_samples = n_samples,
        warmup = warmup,
        thin = thin,
        rng = rng,
        initial = init,
        mala_step_size = mala_step_size,
        mala_target_accept = mala_target_accept,
        mala_adapt_every = mala_adapt_every,
        mala_min_step = mala_min_step,
        mala_max_step = mala_max_step,
        initial_step = initial_step,
        target_accept = target_accept,
        adapt_every = adapt_every,
        min_step = min_step,
        max_step = max_step,
        constraint = resolved_constraint,
        soft_center_scale = scale,
        prox_sum0_lambda = prox_sum0_lambda,
        car_weights = car_weights,
        use_car = use_car,
        car_precision = car_precision,
        progress = progress,
    )
end
