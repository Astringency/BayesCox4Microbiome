mutable struct MALAWorkspace
    grad_u::Vector{Float64}
    grad_prop::Vector{Float64}
    proposal::Vector{Float64}
    noise::Vector{Float64}
    diff::Vector{Float64}
end

function MALAWorkspace(d::Integer)
    d > 0 || throw(ArgumentError("workspace dimension must be positive."))
    return MALAWorkspace(
        Vector{Float64}(undef, d),
        Vector{Float64}(undef, d),
        Vector{Float64}(undef, d),
        Vector{Float64}(undef, d),
        Vector{Float64}(undef, d),
    )
end

function _mala_logq(y::AbstractVector, x::AbstractVector, grad_x::AbstractVector, step_size::Real)
    gamma_L = Float64(step_size)
    diff = y .- x .- gamma_L .* grad_x
    return -0.25 / gamma_L * sum(abs2, diff)
end

function _mala_logq!(
    diff::AbstractVector{Float64},
    y::AbstractVector,
    x::AbstractVector,
    grad_x::AbstractVector,
    step_size::Real,
)
    gamma_L = Float64(step_size)
    length(diff) == length(y) == length(x) == length(grad_x) ||
        throw(DimensionMismatch("MALA log proposal dimensions must match."))
    s = 0.0
    @inbounds for j in eachindex(diff)
        v = Float64(y[j]) - Float64(x[j]) - gamma_L * Float64(grad_x[j])
        diff[j] = v
        s += v * v
    end
    return -0.25 / gamma_L * s
end

function _copy_gradient!(dest::AbstractVector{Float64}, gradient::Function, u::AbstractVector)
    g = gradient(u)
    length(g) == length(dest) || throw(DimensionMismatch("gradient length does not match state."))
    copyto!(dest, g)
    return dest
end

function _mala_step!(
    logpost::Function,
    gradient!::Function,
    u::Vector{Float64},
    current_lp::Float64,
    gamma_L::Float64,
    rng::AbstractRNG,
    work::MALAWorkspace,
)
    d = length(u)
    length(work.proposal) == d || throw(DimensionMismatch("MALA workspace dimension does not match state."))
    gradient!(work.grad_u, u)
    randn!(rng, work.noise)
    scale = sqrt(2 * gamma_L)
    @inbounds for j in 1:d
        work.proposal[j] = u[j] + gamma_L * work.grad_u[j] + scale * work.noise[j]
    end
    prop_lp = logpost(work.proposal)
    if !isfinite(prop_lp)
        return current_lp, false
    end
    gradient!(work.grad_prop, work.proposal)
    log_alpha = prop_lp + _mala_logq!(work.diff, u, work.proposal, work.grad_prop, gamma_L) -
        current_lp - _mala_logq!(work.diff, work.proposal, u, work.grad_u, gamma_L)
    accepted = log(rand(rng)) < log_alpha
    return accepted ? prop_lp : current_lp, accepted
end

function _mala_step(
    logpost::Function,
    gradient::Function,
    u::Vector{Float64},
    current_lp::Float64,
    gamma_L::Float64,
    rng::AbstractRNG,
)
    work = MALAWorkspace(length(u))
    gradient! = (dest, x) -> _copy_gradient!(dest, gradient, x)
    lp_new, accepted = _mala_step!(logpost, gradient!, u, current_lp, gamma_L, rng, work)
    return accepted ? copy(work.proposal) : u, lp_new, accepted
end

function _ula_step!(
    proposal::AbstractVector{Float64},
    gradient!::Function,
    u::Vector{Float64},
    gamma_L::Float64,
    rng::AbstractRNG,
    work::MALAWorkspace,
)
    d = length(u)
    length(proposal) == d || throw(DimensionMismatch("proposal length does not match state."))
    gradient!(work.grad_u, u)
    randn!(rng, work.noise)
    scale = sqrt(2 * gamma_L)
    @inbounds for j in 1:d
        proposal[j] = u[j] + gamma_L * work.grad_u[j] + scale * work.noise[j]
    end
    return proposal
end

function _ula_step(
    gradient::Function,
    u::Vector{Float64},
    gamma_L::Float64,
    rng::AbstractRNG,
)
    work = MALAWorkspace(length(u))
    gradient! = (dest, x) -> _copy_gradient!(dest, gradient, x)
    return copy(_ula_step!(work.proposal, gradient!, u, gamma_L, rng, work))
end

function _full_logposterior_gradient_beta(
    u::AbstractVector,
    data::CoxPreparedData,
    prior::AbstractCoxPrior;
    constraint::Symbol,
    prox_sum0_lambda::Real,
)
    p = size(data.z, 2)
    beta = view(u, 1:p)
    grad = cox_score_gradient(data, beta)
    _constraint_gradient_penalty!(
        grad,
        beta;
        constraint = constraint,
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
        log_tau2_spike = view(u, (p + 1):(2p))
        log_tau2_slab = view(u, (2p + 1):(3p))
        theta = _logistic(u[3p + 1])
        theta = clamp(theta, eps(Float64), 1 - eps(Float64))
        @inbounds for j in 1:p
            b = beta[j]
            tau2_spike = exp(log_tau2_spike[j])
            tau2_slab = exp(log_tau2_slab[j])
            log_spike, log_slab, norm =
                _ssl_component_logpdf(b, theta, log_tau2_spike[j], log_tau2_slab[j])
            prob_spike = exp(log_spike - norm)
            prob_slab = exp(log_slab - norm)
            grad[j] -= b * (prob_spike / tau2_spike + prob_slab / tau2_slab)
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
    mala_precondition::Bool,
    mala_precondition_mode::Symbol,
    mala_precondition_diag::Bool,
    mala_metric_ridge::Real,
    mala_metric_max_cholesky_tries::Integer,
    mala_stepsize_schedule::Symbol,
    mala_sgld_c0::Real,
    mala_sgld_target_accept::Real,
    mala_sgld_accept_low::Real,
    mala_sgld_accept_high::Real,
    mala_sgld_adapt_c0::Bool,
    mala_sgld_adapt_every::Integer,
    mala_sgld_c0_min::Real,
    mala_sgld_c0_max::Real,
    initial_step::Union{Real,AbstractVector{<:Real}},
    target_accept::Real,
    adapt_every::Integer,
    min_step::Real,
    max_step::Real,
    constraint::Symbol,
    prox_sum0_lambda::Real,
    car_weights,
    use_car::Bool,
    car_precision::Real,
    sample_car_strength::Bool,
    car_strength_prior_a::Real,
    car_strength_prior_b::Real,
    car_degree_scale::Symbol,
    car_kappa_init::Real,
    progress::Bool,
    progress_desc::AbstractString = "MALA sampling",
    progress_offset::Integer = 0,
)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive."))
    warmup >= 0 || throw(ArgumentError("warmup must be nonnegative."))
    thin > 0 || throw(ArgumentError("thin must be positive."))
    mala_adapt_every > 0 || throw(ArgumentError("mala_adapt_every must be positive."))
    gamma_L = _check_positive(mala_step_size, "mala_step_size")
    gamma_min = _check_positive(mala_min_step, "mala_min_step")
    gamma_max = _check_positive(mala_max_step, "mala_max_step")
    0 < mala_target_accept < 1 || throw(ArgumentError("mala_target_accept must be in (0, 1)."))
    gamma_max >= gamma_min || throw(ArgumentError("mala_max_step must be at least mala_min_step."))
    metric_ridge = _check_positive(mala_metric_ridge, "mala_metric_ridge")
    mala_metric_max_cholesky_tries > 0 ||
        throw(ArgumentError("mala_metric_max_cholesky_tries must be positive."))
    sgld_c0 = _validate_mala_schedule_options(
        mala_stepsize_schedule,
        mala_sgld_c0,
        mala_sgld_target_accept,
        mala_sgld_accept_low,
        mala_sgld_accept_high,
        mala_sgld_adapt_every,
        mala_sgld_c0_min,
        mala_sgld_c0_max,
    )
    if mala_precondition
        mala_precondition_mode in (:auto, :beta_dense, :beta_diag) ||
            throw(ArgumentError("standard-prior preconditioned MALA supports :auto, :beta_dense, or :beta_diag."))
    end

    p = size(data.z, 2)
    init = collect(Float64, initial)
    names = _state_names(
        prior,
        p;
        sample_car_strength = prior isa Union{HorseshoePrior,RegularizedHorseshoePrior} ? sample_car_strength : false,
    )
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
    beta_grad = u -> _full_logposterior_gradient_beta(
        u,
        data,
        prior;
        constraint = constraint,
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
    last_mala_step = gamma_L
    mala_work = MALAWorkspace(p)
    precond_work = mala_precondition ? PreconditionedMALAWorkspace(p) : nothing
    metric_diag = mala_precondition_diag || mala_precondition_mode == :beta_diag
    n_obs = size(data.z, 1)
    prog = _progress_meter(progress, total_iter, "MALA sampling"; progress_desc = progress_desc, progress_offset = progress_offset)

    for iter in 1:total_iter
        mala_step = mala_stepsize_schedule == :adaptive_sgld ?
            _mala_sgld_epsilon(sgld_c0, iter, n_obs, p, gamma_min, gamma_max) :
            gamma_L
        last_mala_step = mala_step
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
        beta_state = collect(view(u, 1:p))
        beta_lp, accepted =
            if mala_precondition
                beta_gradient_current! = (grad, b) -> begin
                    copyto!(grad, beta_gradient_current(b))
                    return grad
                end
                beta_metric_current! = (G, b) -> begin
                    uu = copy(u)
                    uu[1:p] .= b
                    return standard_observed_information_beta!(
                        G,
                        data,
                        prior,
                        uu;
                        constraint = constraint,
                        prox_sum0_lambda = prox_sum0_lambda,
                        ridge = 0.0,
                        diag_only = metric_diag,
                    )
                end
                _precond_mala_step!(
                    beta_logpost_current,
                    beta_gradient_current!,
                    beta_metric_current!,
                    beta_state,
                    current_lp,
                    Float64(mala_step),
                    rng,
                    precond_work;
                    ridge = metric_ridge,
                    max_cholesky_tries = Int(mala_metric_max_cholesky_tries),
                )
            else
                _mala_step!(
                    beta_logpost_current,
                    (grad, b) -> _copy_gradient!(grad, beta_gradient_current, b),
                    beta_state,
                    current_lp,
                    Float64(mala_step),
                    rng,
                    mala_work,
                )
            end
        if accepted
            u[1:p] .= mala_precondition ? precond_work.proposal : mala_work.proposal
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

        if mala_stepsize_schedule == :adaptive_mcmc && iter <= warmup && iter % mala_adapt_every == 0
            window_rate = accept_window / mala_adapt_every
            gamma_L = clamp(gamma_L * exp(clamp(window_rate - mala_target_accept, -0.5, 0.5)), gamma_min, gamma_max)
            accept_window = 0
        elseif mala_stepsize_schedule == :adaptive_sgld && iter <= warmup && iter % mala_sgld_adapt_every == 0
            window_rate = accept_window / mala_sgld_adapt_every
            if mala_sgld_adapt_c0
                sgld_c0 = _adapt_mala_sgld_c0(
                    sgld_c0,
                    window_rate,
                    mala_sgld_accept_low,
                    mala_sgld_accept_high,
                    mala_sgld_c0_min,
                    mala_sgld_c0_max,
                )
            end
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
        _progress_next!(prog)
    end
    _progress_finish!(prog)

    sample_iters = n_samples * thin
    warmup_rate = warmup == 0 ? NaN : accept_warmup / warmup
    sample_rate = sample_iters == 0 ? NaN : accept_sample / sample_iters
    beta_final_step = mala_stepsize_schedule == :adaptive_sgld ? last_mala_step : gamma_L
    final_step = vcat(fill(beta_final_step, p), hyper_step)
    return MHSamples(draws, lps, names, sample_rate, warmup_rate, final_step)
end

function _hybrid_beta_ula(
    data::CoxPreparedData,
    prior::AbstractCoxPrior;
    n_samples::Integer,
    warmup::Integer,
    thin::Integer,
    rng::AbstractRNG,
    initial,
    mala_step_size::Real,
    initial_step::Union{Real,AbstractVector{<:Real}},
    constraint::Symbol,
    prox_sum0_lambda::Real,
    car_weights,
    use_car::Bool,
    car_precision::Real,
    sample_car_strength::Bool,
    car_strength_prior_a::Real,
    car_strength_prior_b::Real,
    car_degree_scale::Symbol,
    car_kappa_init::Real,
    progress::Bool,
    progress_desc::AbstractString = "ULA sampling",
    progress_offset::Integer = 0,
)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive."))
    warmup >= 0 || throw(ArgumentError("warmup must be nonnegative."))
    thin > 0 || throw(ArgumentError("thin must be positive."))
    gamma_L = _check_positive(mala_step_size, "mala_step_size")

    p = size(data.z, 2)
    init = collect(Float64, initial)
    names = _state_names(
        prior,
        p;
        sample_car_strength = prior isa Union{HorseshoePrior,RegularizedHorseshoePrior} ? sample_car_strength : false,
    )
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

    logpost = u -> logposterior(
        u,
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
    beta_grad = u -> _full_logposterior_gradient_beta(
        u,
        data,
        prior;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
    )

    total_iter = warmup + n_samples * thin
    draws = Matrix{Float64}(undef, n_samples, d)
    lps = Vector{Float64}(undef, n_samples)
    u = init
    current_lp = logpost(u)
    isfinite(current_lp) || throw(ArgumentError("initial state has non-finite log posterior."))

    hyper_accept_warmup = 0
    hyper_accept_sample = 0
    saved = 0
    prog = _progress_meter(progress, total_iter, "ULA sampling"; progress_desc = progress_desc, progress_offset = progress_offset)

    for iter in 1:total_iter
        beta_gradient_current = b -> begin
            uu = copy(u)
            uu[1:p] .= b
            return beta_grad(uu)
        end
        beta_new = _ula_step(beta_gradient_current, collect(view(u, 1:p)), gamma_L, rng)
        prop = copy(u)
        prop[1:p] .= beta_new
        prop_lp = logpost(prop)
        if isfinite(prop_lp)
            u = prop
            current_lp = prop_lp
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
                else
                    hyper_accept_sample += 1
                end
            end
        end

        if iter > warmup && (iter - warmup) % thin == 0
            saved += 1
            @views draws[saved, :] .= u
            lps[saved] = current_lp
        end
        _progress_next!(prog)
    end
    _progress_finish!(prog)

    sample_iters = n_samples * thin
    warmup_rate = hdim == 0 || warmup == 0 ? NaN : hyper_accept_warmup / warmup
    sample_rate = hdim == 0 || sample_iters == 0 ? NaN : hyper_accept_sample / sample_iters
    return MHSamples(draws, lps, names, sample_rate, warmup_rate, vcat(fill(gamma_L, p), hyper_step))
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
    mala_precondition::Bool,
    mala_precondition_mode::Symbol,
    mala_precondition_diag::Bool,
    mala_metric_ridge::Real,
    mala_metric_max_cholesky_tries::Integer,
    mala_stepsize_schedule::Symbol,
    mala_sgld_c0::Real,
    mala_sgld_target_accept::Real,
    mala_sgld_accept_low::Real,
    mala_sgld_accept_high::Real,
    mala_sgld_adapt_c0::Bool,
    mala_sgld_adapt_every::Integer,
    mala_sgld_c0_min::Real,
    mala_sgld_c0_max::Real,
    constraint::Symbol,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    alpha_prior_kind = :invgamma,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
    init::Symbol = :zero,
    progress::Bool,
    progress_desc::AbstractString = "MALA sampling",
    progress_offset::Integer = 0,
)
    p = size(data.z, 2)
    init_vec = if initial !== nothing
        collect(Float64, initial)
    elseif init == :rand
        vcat(_default_unconstrained_init(rng, p), [0.0])  # η=0 → α=1 avoids gradient explosion with small prox_l1_lambda
    else
        vcat(zeros(Float64, p), [0.0])
    end
    length(init_vec) == p + 1 || throw(DimensionMismatch("initial length must be p + 1 for prox LASSO MALA."))
    names = vcat(["beta[$j]" for j in 1:p], ["eta"])
    logpost = u -> _lasso_prox_logposterior(
        u,
        data,
        prior;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        alpha_prior_kind = alpha_prior_kind,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    gradient = u -> _lasso_prox_gradient(
        u,
        data;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        alpha_prior_kind = alpha_prior_kind,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    work = ProxGradientWorkspace(data)
    gradient! = (grad, u) -> _lasso_prox_gradient!(
        grad,
        u,
        data,
        work;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        alpha_prior_kind = alpha_prior_kind,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    if mala_precondition
        mode = mala_precondition_mode == :auto ? :full_dense : mala_precondition_mode
        mode == :full_dense ||
            throw(ArgumentError("prox-lasso preconditioned MALA requires mala_precondition_mode=:auto or :full_dense."))
    end
    metric! = (G, u) -> prox_lasso_observed_information!(
        G,
        u,
        data,
        prior;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        alpha_prior_kind = alpha_prior_kind,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
        ridge = 0.0,
        diag_only = mala_precondition_diag,
    )
    return _run_full_mala(logpost, gradient, init_vec, names; n_samples = n_samples, warmup = warmup, thin = thin, rng = rng,
        mala_step_size = mala_step_size, mala_target_accept = mala_target_accept, mala_adapt_every = mala_adapt_every,
        mala_min_step = mala_min_step, mala_max_step = mala_max_step, mala_precondition = mala_precondition,
        mala_metric_ridge = mala_metric_ridge, mala_metric_max_cholesky_tries = mala_metric_max_cholesky_tries,
        mala_stepsize_schedule = mala_stepsize_schedule, mala_sgld_c0 = mala_sgld_c0,
        mala_sgld_target_accept = mala_sgld_target_accept, mala_sgld_accept_low = mala_sgld_accept_low,
        mala_sgld_accept_high = mala_sgld_accept_high, mala_sgld_adapt_c0 = mala_sgld_adapt_c0,
        mala_sgld_adapt_every = mala_sgld_adapt_every, mala_sgld_c0_min = mala_sgld_c0_min,
        mala_sgld_c0_max = mala_sgld_c0_max, n_obs = size(data.z, 1), d_block = p + 1,
        metric! = metric!, progress = progress, progress_desc = progress_desc,
        progress_offset = progress_offset, gradient! = gradient!)
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
    mala_precondition::Bool = false,
    mala_metric_ridge::Real = 1e-6,
    mala_metric_max_cholesky_tries::Integer = 8,
    mala_stepsize_schedule::Symbol = :adaptive_mcmc,
    mala_sgld_c0::Real = 1.0,
    mala_sgld_target_accept::Real = 0.5,
    mala_sgld_accept_low::Real = 0.40,
    mala_sgld_accept_high::Real = 0.60,
    mala_sgld_adapt_c0::Bool = true,
    mala_sgld_adapt_every::Integer = mala_adapt_every,
    mala_sgld_c0_min::Real = 1e-12,
    mala_sgld_c0_max::Real = 1e12,
    n_obs::Integer = 1,
    d_block::Integer = length(init),
    metric! = nothing,
    progress::Bool,
    progress_desc::AbstractString = "MALA sampling",
    progress_offset::Integer = 0,
    gradient! = nothing,
)
    gamma_L = _check_positive(mala_step_size, "mala_step_size")
    gamma_min = _check_positive(mala_min_step, "mala_min_step")
    gamma_max = _check_positive(mala_max_step, "mala_max_step")
    mala_adapt_every > 0 || throw(ArgumentError("mala_adapt_every must be positive."))
    gamma_max >= gamma_min || throw(ArgumentError("mala_max_step must be at least mala_min_step."))
    metric_ridge = _check_positive(mala_metric_ridge, "mala_metric_ridge")
    mala_metric_max_cholesky_tries > 0 ||
        throw(ArgumentError("mala_metric_max_cholesky_tries must be positive."))
    sgld_c0 = _validate_mala_schedule_options(
        mala_stepsize_schedule,
        mala_sgld_c0,
        mala_sgld_target_accept,
        mala_sgld_accept_low,
        mala_sgld_accept_high,
        mala_sgld_adapt_every,
        mala_sgld_c0_min,
        mala_sgld_c0_max,
    )
    total_iter = warmup + n_samples * thin
    u = collect(Float64, init)
    d = length(u)
    d_block > 0 || throw(ArgumentError("d_block must be positive."))
    n_obs > 0 || throw(ArgumentError("n_obs must be positive."))
    mala_precondition && metric! === nothing &&
        throw(ArgumentError("metric! is required when mala_precondition=true."))
    draws = Matrix{Float64}(undef, n_samples, d)
    lps = Vector{Float64}(undef, n_samples)
    current_lp = logpost(u)
    isfinite(current_lp) || throw(ArgumentError("initial state has non-finite log posterior."))
    work = MALAWorkspace(d)
    precond_work = mala_precondition ? PreconditionedMALAWorkspace(d) : nothing
    gradient_bang = gradient! === nothing ? ((dest, x) -> _copy_gradient!(dest, gradient, x)) : gradient!
    accept_warmup = 0
    accept_sample = 0
    accept_window = 0
    saved = 0
    last_mala_step = gamma_L
    prog = _progress_meter(progress, total_iter, "MALA sampling"; progress_desc = progress_desc, progress_offset = progress_offset)
    for iter in 1:total_iter
        mala_step = mala_stepsize_schedule == :adaptive_sgld ?
            _mala_sgld_epsilon(sgld_c0, iter, n_obs, d_block, gamma_min, gamma_max) :
            gamma_L
        last_mala_step = mala_step
        lp_new, accepted =
            if mala_precondition
                _precond_mala_step!(
                    logpost,
                    gradient_bang,
                    metric!,
                    u,
                    current_lp,
                    Float64(mala_step),
                    rng,
                    precond_work;
                    ridge = metric_ridge,
                    max_cholesky_tries = Int(mala_metric_max_cholesky_tries),
                )
            else
                _mala_step!(logpost, gradient_bang, u, current_lp, Float64(mala_step), rng, work)
            end
        if accepted
            copyto!(u, mala_precondition ? precond_work.proposal : work.proposal)
            current_lp = lp_new
            if iter <= warmup
                accept_warmup += 1
                accept_window += 1
            else
                accept_sample += 1
            end
        end
        if mala_stepsize_schedule == :adaptive_mcmc && iter <= warmup && iter % mala_adapt_every == 0
            window_rate = accept_window / mala_adapt_every
            gamma_L = clamp(gamma_L * exp(clamp(window_rate - mala_target_accept, -0.5, 0.5)), gamma_min, gamma_max)
            accept_window = 0
        elseif mala_stepsize_schedule == :adaptive_sgld && iter <= warmup && iter % mala_sgld_adapt_every == 0
            window_rate = accept_window / mala_sgld_adapt_every
            if mala_sgld_adapt_c0
                sgld_c0 = _adapt_mala_sgld_c0(
                    sgld_c0,
                    window_rate,
                    mala_sgld_accept_low,
                    mala_sgld_accept_high,
                    mala_sgld_c0_min,
                    mala_sgld_c0_max,
                )
            end
            accept_window = 0
        elseif iter > warmup && (iter - warmup) % thin == 0
            saved += 1
            @views draws[saved, :] .= u
            lps[saved] = current_lp
        end
        _progress_next!(prog)
    end
    _progress_finish!(prog)
    sample_iters = n_samples * thin
    return MHSamples(
        draws,
        lps,
        names,
        sample_iters == 0 ? NaN : accept_sample / sample_iters,
        warmup == 0 ? NaN : accept_warmup / warmup,
        fill(mala_stepsize_schedule == :adaptive_sgld ? last_mala_step : gamma_L, d),
    )
end

function _run_full_ula(
    logpost::Function,
    gradient::Function,
    init::AbstractVector,
    names::Vector{String};
    n_samples::Integer,
    warmup::Integer,
    thin::Integer,
    rng::AbstractRNG,
    mala_step_size::Real,
    progress::Bool,
    progress_desc::AbstractString = "ULA sampling",
    progress_offset::Integer = 0,
    gradient! = nothing,
)
    gamma_L = _check_positive(mala_step_size, "mala_step_size")
    total_iter = warmup + n_samples * thin
    u = collect(Float64, init)
    d = length(u)
    draws = Matrix{Float64}(undef, n_samples, d)
    lps = Vector{Float64}(undef, n_samples)
    current_lp = logpost(u)
    isfinite(current_lp) || throw(ArgumentError("initial state has non-finite log posterior."))
    work = MALAWorkspace(d)
    gradient_bang = gradient! === nothing ? ((dest, x) -> _copy_gradient!(dest, gradient, x)) : gradient!
    saved = 0
    prog = _progress_meter(progress, total_iter, "ULA sampling"; progress_desc = progress_desc, progress_offset = progress_offset)
    for iter in 1:total_iter
        prop = _ula_step!(work.proposal, gradient_bang, u, gamma_L, rng, work)
        prop_lp = logpost(prop)
        if isfinite(prop_lp)
            copyto!(u, prop)
            current_lp = prop_lp
        end
        if iter > warmup && (iter - warmup) % thin == 0
            saved += 1
            @views draws[saved, :] .= u
            lps[saved] = current_lp
        end
        _progress_next!(prog)
    end
    _progress_finish!(prog)
    return MHSamples(draws, lps, names, NaN, NaN, fill(gamma_L, d))
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
    mala_precondition::Bool,
    mala_precondition_mode::Symbol,
    mala_precondition_diag::Bool,
    mala_metric_ridge::Real,
    mala_metric_max_cholesky_tries::Integer,
    mala_stepsize_schedule::Symbol,
    mala_sgld_c0::Real,
    mala_sgld_target_accept::Real,
    mala_sgld_accept_low::Real,
    mala_sgld_accept_high::Real,
    mala_sgld_adapt_c0::Bool,
    mala_sgld_adapt_every::Integer,
    mala_sgld_c0_min::Real,
    mala_sgld_c0_max::Real,
    constraint::Symbol,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    alpha_prior_kind::Symbol,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
    init::Symbol = :zero,
    progress::Bool,
    progress_desc::AbstractString = "SSLASSO prox MALA",
    progress_offset::Integer = 0,
)
    # Conditional SSLASSO proximal epigraph sampler: gamma is discrete and is
    # updated by Bernoulli Gibbs, while MALA only targets continuous beta/eta.
    # This is distinct from the marginal mixture-Laplace SSLASSO target.
    p = size(data.z, 2)
    u = if initial !== nothing
        collect(Float64, initial)
    elseif init == :rand
        vcat(_default_unconstrained_init(rng, p), [0.0])  # η=0 → α=1 avoids gradient explosion with small prox_l1_lambda
    else
        vcat(zeros(Float64, p), [0.0])
    end
    length(u) == p + 1 || throw(DimensionMismatch("initial length must be p + 1 for prox SSLASSO MALA."))
    gamma = trues(p)
    theta = 0.5
    weights = _ssl_weights(prior, gamma)
    spike_weight = inv(prior.spike_scale)
    slab_weight = inv(prior.slab_scale)
    log_weight_odds = log(slab_weight) - log(spike_weight)
    names = vcat(["beta[$j]" for j in 1:p], ["eta"], ["gamma[$j]" for j in 1:p], ["theta"])
    d = length(names)
    draws = Matrix{Float64}(undef, n_samples, d)
    lps = Vector{Float64}(undef, n_samples)
    gamma_L = _check_positive(mala_step_size, "mala_step_size")
    gamma_min = _check_positive(mala_min_step, "mala_min_step")
    gamma_max = _check_positive(mala_max_step, "mala_max_step")
    mala_adapt_every > 0 || throw(ArgumentError("mala_adapt_every must be positive."))
    gamma_max >= gamma_min || throw(ArgumentError("mala_max_step must be at least mala_min_step."))
    metric_ridge = _check_positive(mala_metric_ridge, "mala_metric_ridge")
    mala_metric_max_cholesky_tries > 0 ||
        throw(ArgumentError("mala_metric_max_cholesky_tries must be positive."))
    sgld_c0 = _validate_mala_schedule_options(
        mala_stepsize_schedule,
        mala_sgld_c0,
        mala_sgld_target_accept,
        mala_sgld_accept_low,
        mala_sgld_accept_high,
        mala_sgld_adapt_every,
        mala_sgld_c0_min,
        mala_sgld_c0_max,
    )
    if mala_precondition
        mode = mala_precondition_mode == :auto ? :full_dense : mala_precondition_mode
        mode == :full_dense ||
            throw(ArgumentError("prox-sslasso preconditioned MALA requires mala_precondition_mode=:auto or :full_dense."))
    end
    total_iter = warmup + n_samples * thin
    logpost = uu -> _ssl_prox_logposterior(
        uu,
        data,
        prior,
        gamma,
        theta;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        alpha_prior_kind = alpha_prior_kind,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    current_lp = logpost(u)
    isfinite(current_lp) || throw(ArgumentError("initial state has non-finite log posterior."))
    mala_work = MALAWorkspace(p + 1)
    precond_work = mala_precondition ? PreconditionedMALAWorkspace(p + 1) : nothing
    prox_work = ProxGradientWorkspace(data)
    gradient_current! = (grad, uu) -> _ssl_prox_gradient!(
        grad,
        uu,
        data,
        prior,
        gamma,
        prox_work;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        alpha_prior_kind = alpha_prior_kind,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    accept_warmup = 0
    accept_sample = 0
    accept_window = 0
    saved = 0
    last_mala_step = gamma_L
    prog = _progress_meter(progress, total_iter, "SSLASSO prox MALA"; progress_desc = progress_desc, progress_offset = progress_offset)
    for iter in 1:total_iter
        mala_step = mala_stepsize_schedule == :adaptive_sgld ?
            _mala_sgld_epsilon(sgld_c0, iter, size(data.z, 1), p + 1, gamma_min, gamma_max) :
            gamma_L
        last_mala_step = mala_step
        logpost_current = uu -> _ssl_prox_logposterior(
            uu,
            data,
            prior,
            gamma,
            theta;
            constraint = constraint,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = alpha_prior_shape,
            alpha_prior_scale = alpha_prior_scale,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
        )
        metric_current! = (G, uu) -> prox_ssl_observed_information!(
            G,
            uu,
            data,
            prior,
            gamma;
            constraint = constraint,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = alpha_prior_shape,
            alpha_prior_scale = alpha_prior_scale,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
            ridge = 0.0,
            diag_only = mala_precondition_diag,
        )
        lp_new, accepted =
            if mala_precondition
                _precond_mala_step!(
                    logpost_current,
                    gradient_current!,
                    metric_current!,
                    u,
                    current_lp,
                    Float64(mala_step),
                    rng,
                    precond_work;
                    ridge = metric_ridge,
                    max_cholesky_tries = Int(mala_metric_max_cholesky_tries),
                )
            else
                _mala_step!(logpost_current, gradient_current!, u, current_lp, Float64(mala_step), rng, mala_work)
            end
        if accepted
            copyto!(u, mala_precondition ? precond_work.proposal : mala_work.proposal)
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
            weights[j] = spike_weight
            h0 = weighted_l1_moreau(beta, alpha, weights, prox_l1_lambda;
                prox_root_tol = prox_root_tol, prox_root_maxiter = prox_root_maxiter)
            gamma[j] = true
            weights[j] = slab_weight
            h1 = weighted_l1_moreau(beta, alpha, weights, prox_l1_lambda;
                prox_root_tol = prox_root_tol, prox_root_maxiter = prox_root_maxiter)
            logit_p = log(theta_clamped) - log1p(-theta_clamped) + log_weight_odds - (h1 - h0)
            gamma[j] = rand(rng) < _logistic(logit_p)
            weights[j] = gamma[j] ? slab_weight : spike_weight
        end
        theta = rand(rng, Beta(prior.theta_a + count(gamma), prior.theta_b + p - count(gamma)))
        current_lp = logpost_current(u)

        if mala_stepsize_schedule == :adaptive_mcmc && iter <= warmup && iter % mala_adapt_every == 0
            window_rate = accept_window / mala_adapt_every
            gamma_L = clamp(gamma_L * exp(clamp(window_rate - mala_target_accept, -0.5, 0.5)), gamma_min, gamma_max)
            accept_window = 0
        elseif mala_stepsize_schedule == :adaptive_sgld && iter <= warmup && iter % mala_sgld_adapt_every == 0
            window_rate = accept_window / mala_sgld_adapt_every
            if mala_sgld_adapt_c0
                sgld_c0 = _adapt_mala_sgld_c0(
                    sgld_c0,
                    window_rate,
                    mala_sgld_accept_low,
                    mala_sgld_accept_high,
                    mala_sgld_c0_min,
                    mala_sgld_c0_max,
                )
            end
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
                prox_sum0_lambda = prox_sum0_lambda,
                prox_l1_lambda = prox_l1_lambda,
                alpha_prior_shape = alpha_prior_shape,
                alpha_prior_scale = alpha_prior_scale,
                prox_root_tol = prox_root_tol,
                prox_root_maxiter = prox_root_maxiter,
            )
        end
        _progress_next!(prog)
    end
    _progress_finish!(prog)
    sample_iters = n_samples * thin
    final_mala_step = mala_stepsize_schedule == :adaptive_sgld ? last_mala_step : gamma_L
    return MHSamples(draws, lps, names, sample_iters == 0 ? NaN : accept_sample / sample_iters,
        warmup == 0 ? NaN : accept_warmup / warmup, fill(final_mala_step, d))
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
    mala_precondition_mode::Symbol = :auto,
    mala_precondition_diag::Bool = false,
    mala_metric_ridge::Real = 1e-6,
    mala_metric_max_cholesky_tries::Integer = 8,
    mala_stepsize_schedule::Symbol = :adaptive_mcmc,
    mala_sgld_c0::Real = 1.0,
    mala_sgld_target_accept::Real = 0.5,
    mala_sgld_accept_low::Real = 0.40,
    mala_sgld_accept_high::Real = 0.60,
    mala_sgld_adapt_c0::Bool = true,
    mala_sgld_adapt_every::Integer = mala_adapt_every,
    mala_sgld_c0_min::Real = 1e-12,
    mala_sgld_c0_max::Real = 1e12,
    constraint::Symbol = :soft_center,
    prox_sum0_lambda::Real = 1e-3,
    prox_l1_lambda::Real = 1e-3,
    alpha_prior_shape = nothing,
    alpha_prior_scale = nothing,
    alpha_prior_kind = :invgamma,
    prox_root_tol::Real = 1e-10,
    prox_root_maxiter::Integer = 100,
    init::Symbol = :zero,
    car_weights = nothing,
    use_car::Bool = car_weights !== nothing,
    car_precision::Real = 1.0,
    sample_car_strength::Bool = false,
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_degree_scale::Symbol = :mean_active,
    car_kappa_init::Real = 0.5,
    progress::Bool = true,
    progress_desc::AbstractString = "MALA sampling",
    progress_offset::Integer = 0,
)
    _validate_prior(prior)
    _validate_mala_schedule_options(
        mala_stepsize_schedule,
        mala_sgld_c0,
        mala_sgld_target_accept,
        mala_sgld_accept_low,
        mala_sgld_accept_high,
        mala_sgld_adapt_every,
        mala_sgld_c0_min,
        mala_sgld_c0_max,
    )
    _check_positive(mala_metric_ridge, "mala_metric_ridge")
    mala_metric_max_cholesky_tries > 0 ||
        throw(ArgumentError("mala_metric_max_cholesky_tries must be positive."))
    p = size(data.z, 2)
    n_events = count(!=(0), data.event)
    resolved_constraint = _resolve_constraint(constraint)
    effective_car_weights = use_car ? car_weights : nothing
    if sample_car_strength
        _validate_adaptive_car_options(
            effective_car_weights,
            car_strength_prior_a,
            car_strength_prior_b,
            car_kappa_init,
        )
    end
    _check_positive(prox_sum0_lambda, "prox_sum0_lambda")
    _check_positive(prox_l1_lambda, "prox_l1_lambda")
    _check_positive(prox_root_tol, "prox_root_tol")
    prox_root_maxiter > 0 || throw(ArgumentError("prox_root_maxiter must be positive."))
    a_shape = Float64(something(alpha_prior_shape, prior isa BayesianLassoPrior ? prior.lambda2_shape : 1.0))
    a_scale = Float64(something(alpha_prior_scale, prior isa BayesianLassoPrior ? prior.lambda2_rate : 1.0))
    a_kind = _normalize_alpha_prior_kind(alpha_prior_kind)
    _check_positive(a_shape, "alpha_prior_shape")
    _check_positive(a_scale, "alpha_prior_scale")
    use_proximal_prior = _is_proximal_l1_prior(prior)

    if prior isa BayesianLassoPrior && use_proximal_prior
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
            mala_precondition = mala_precondition,
            mala_precondition_mode = mala_precondition_mode,
            mala_precondition_diag = mala_precondition_diag,
            mala_metric_ridge = mala_metric_ridge,
            mala_metric_max_cholesky_tries = mala_metric_max_cholesky_tries,
            mala_stepsize_schedule = mala_stepsize_schedule,
            mala_sgld_c0 = mala_sgld_c0,
            mala_sgld_target_accept = mala_sgld_target_accept,
            mala_sgld_accept_low = mala_sgld_accept_low,
            mala_sgld_accept_high = mala_sgld_accept_high,
            mala_sgld_adapt_c0 = mala_sgld_adapt_c0,
            mala_sgld_adapt_every = mala_sgld_adapt_every,
            mala_sgld_c0_min = mala_sgld_c0_min,
            mala_sgld_c0_max = mala_sgld_c0_max,
            constraint = resolved_constraint,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = a_shape,
            alpha_prior_scale = a_scale,
            alpha_prior_kind = a_kind,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
            init = init,
            progress = progress,
            progress_desc = progress_desc,
            progress_offset = progress_offset,
        )
    elseif prior isa SpikeSlabLassoPrior && use_proximal_prior
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
            mala_precondition = mala_precondition,
            mala_precondition_mode = mala_precondition_mode,
            mala_precondition_diag = mala_precondition_diag,
            mala_metric_ridge = mala_metric_ridge,
            mala_metric_max_cholesky_tries = mala_metric_max_cholesky_tries,
            mala_stepsize_schedule = mala_stepsize_schedule,
            mala_sgld_c0 = mala_sgld_c0,
            mala_sgld_target_accept = mala_sgld_target_accept,
            mala_sgld_accept_low = mala_sgld_accept_low,
            mala_sgld_accept_high = mala_sgld_accept_high,
            mala_sgld_adapt_c0 = mala_sgld_adapt_c0,
            mala_sgld_adapt_every = mala_sgld_adapt_every,
            mala_sgld_c0_min = mala_sgld_c0_min,
            mala_sgld_c0_max = mala_sgld_c0_max,
            constraint = resolved_constraint,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = a_shape,
            alpha_prior_scale = a_scale,
            alpha_prior_kind = a_kind,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
            init = init,
            progress = progress,
            progress_desc = progress_desc,
            progress_offset = progress_offset,
        )
    end

    init_vec =
        if initial !== nothing
            collect(Float64, initial)
        elseif prior isa RegularizedHorseshoePrior
            initial_state(
                prior,
                p,
                n_events;
                n_obs = size(data.z, 1),
                init = init,
                car_weights = effective_car_weights,
                sample_car_strength = sample_car_strength,
                car_strength_prior_a = car_strength_prior_a,
                car_strength_prior_b = car_strength_prior_b,
                car_kappa_init = car_kappa_init,
            )
        else
            initial_state(
                prior,
                p,
                n_events;
                init = init,
                car_weights = effective_car_weights,
                sample_car_strength = prior isa Union{HorseshoePrior,RegularizedHorseshoePrior} ? sample_car_strength : false,
                car_strength_prior_a = car_strength_prior_a,
                car_strength_prior_b = car_strength_prior_b,
                car_kappa_init = car_kappa_init,
            )
        end
    return _hybrid_beta_mala(
        data,
        prior;
        n_samples = n_samples,
        warmup = warmup,
        thin = thin,
        rng = rng,
        initial = init_vec,
        mala_step_size = mala_step_size,
        mala_target_accept = mala_target_accept,
        mala_adapt_every = mala_adapt_every,
        mala_min_step = mala_min_step,
        mala_max_step = mala_max_step,
        mala_precondition = mala_precondition,
        mala_precondition_mode = mala_precondition_mode,
        mala_precondition_diag = mala_precondition_diag,
        mala_metric_ridge = mala_metric_ridge,
        mala_metric_max_cholesky_tries = mala_metric_max_cholesky_tries,
        mala_stepsize_schedule = mala_stepsize_schedule,
        mala_sgld_c0 = mala_sgld_c0,
        mala_sgld_target_accept = mala_sgld_target_accept,
        mala_sgld_accept_low = mala_sgld_accept_low,
        mala_sgld_accept_high = mala_sgld_accept_high,
        mala_sgld_adapt_c0 = mala_sgld_adapt_c0,
        mala_sgld_adapt_every = mala_sgld_adapt_every,
        mala_sgld_c0_min = mala_sgld_c0_min,
        mala_sgld_c0_max = mala_sgld_c0_max,
        initial_step = initial_step,
        target_accept = target_accept,
        adapt_every = adapt_every,
        min_step = min_step,
        max_step = max_step,
        constraint = resolved_constraint,
        prox_sum0_lambda = prox_sum0_lambda,
        car_weights = car_weights,
        use_car = use_car,
        car_precision = car_precision,
        sample_car_strength = sample_car_strength,
        car_strength_prior_a = car_strength_prior_a,
        car_strength_prior_b = car_strength_prior_b,
        car_degree_scale = car_degree_scale,
        car_kappa_init = car_kappa_init,
        progress = progress,
        progress_desc = progress_desc,
        progress_offset = progress_offset,
    )
end

function _sample_ula(
    data::CoxPreparedData,
    prior::AbstractCoxPrior;
    n_samples::Integer = 5000,
    warmup::Integer = 2000,
    thin::Integer = 1,
    rng::AbstractRNG = Random.default_rng(),
    initial = nothing,
    initial_step::Union{Real,AbstractVector{<:Real}} = 0.05,
    mala_step_size::Real = 1e-4,
    constraint::Symbol = :soft_center,
    prox_sum0_lambda::Real = 1e-3,
    prox_l1_lambda::Real = 1e-3,
    alpha_prior_shape = nothing,
    alpha_prior_scale = nothing,
    alpha_prior_kind = :invgamma,
    prox_root_tol::Real = 1e-10,
    prox_root_maxiter::Integer = 100,
    init::Symbol = :zero,
    car_weights = nothing,
    use_car::Bool = car_weights !== nothing,
    car_precision::Real = 1.0,
    sample_car_strength::Bool = false,
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_degree_scale::Symbol = :mean_active,
    car_kappa_init::Real = 0.5,
    progress::Bool = true,
    progress_desc::AbstractString = "ULA sampling",
    progress_offset::Integer = 0,
)
    _validate_prior(prior)
    p = size(data.z, 2)
    n_events = count(!=(0), data.event)
    resolved_constraint = _resolve_constraint(constraint)
    effective_car_weights = use_car ? car_weights : nothing
    if sample_car_strength
        _validate_adaptive_car_options(
            effective_car_weights,
            car_strength_prior_a,
            car_strength_prior_b,
            car_kappa_init,
        )
    end
    _check_positive(prox_sum0_lambda, "prox_sum0_lambda")
    _check_positive(prox_l1_lambda, "prox_l1_lambda")
    _check_positive(prox_root_tol, "prox_root_tol")
    prox_root_maxiter > 0 || throw(ArgumentError("prox_root_maxiter must be positive."))
    a_shape = Float64(something(alpha_prior_shape, prior isa BayesianLassoPrior ? prior.lambda2_shape : 1.0))
    a_scale = Float64(something(alpha_prior_scale, prior isa BayesianLassoPrior ? prior.lambda2_rate : 1.0))
    a_kind = _normalize_alpha_prior_kind(alpha_prior_kind)
    use_proximal_prior = _is_proximal_l1_prior(prior)

    if prior isa BayesianLassoPrior && use_proximal_prior
        init_vec = if initial !== nothing
            collect(Float64, initial)
        elseif init == :rand
            vcat(_default_unconstrained_init(rng, p), [0.0])  # η=0 → α=1 avoids gradient explosion with small prox_l1_lambda
        else
            vcat(zeros(Float64, p), [0.0])
        end
        names = vcat(["beta[$j]" for j in 1:p], ["eta"])
        logpost = u -> _lasso_prox_logposterior(
            u,
            data,
            prior;
            constraint = resolved_constraint,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = a_shape,
            alpha_prior_scale = a_scale,
            alpha_prior_kind = a_kind,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
        )
        gradient = u -> _lasso_prox_gradient(
            u,
            data;
            constraint = resolved_constraint,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = a_shape,
            alpha_prior_scale = a_scale,
            alpha_prior_kind = a_kind,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
        )
        work = ProxGradientWorkspace(data)
        gradient! = (grad, u) -> _lasso_prox_gradient!(
            grad,
            u,
            data,
            work;
            constraint = resolved_constraint,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = a_shape,
            alpha_prior_scale = a_scale,
            alpha_prior_kind = a_kind,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
        )
        return _run_full_ula(logpost, gradient, init_vec, names; n_samples = n_samples, warmup = warmup, thin = thin,
            rng = rng, mala_step_size = mala_step_size, progress = progress,
            progress_desc = progress_desc, progress_offset = progress_offset, gradient! = gradient!)
    elseif prior isa SpikeSlabLassoPrior && use_proximal_prior
        return _sample_ssl_prox_ula(
            data,
            prior;
            n_samples = n_samples,
            warmup = warmup,
            thin = thin,
            rng = rng,
            initial = initial,
            mala_step_size = mala_step_size,
            constraint = resolved_constraint,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = a_shape,
            alpha_prior_scale = a_scale,
            alpha_prior_kind = a_kind,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
            init = init,
            progress = progress,
            progress_desc = progress_desc,
            progress_offset = progress_offset,
        )
    end

    init_vec =
        if initial !== nothing
            collect(Float64, initial)
        elseif prior isa RegularizedHorseshoePrior
            initial_state(
                prior,
                p,
                n_events;
                n_obs = size(data.z, 1),
                init = init,
                car_weights = effective_car_weights,
                sample_car_strength = sample_car_strength,
                car_strength_prior_a = car_strength_prior_a,
                car_strength_prior_b = car_strength_prior_b,
                car_kappa_init = car_kappa_init,
            )
        else
            initial_state(
                prior,
                p,
                n_events;
                init = init,
                car_weights = effective_car_weights,
                sample_car_strength = prior isa Union{HorseshoePrior,RegularizedHorseshoePrior} ? sample_car_strength : false,
                car_strength_prior_a = car_strength_prior_a,
                car_strength_prior_b = car_strength_prior_b,
                car_kappa_init = car_kappa_init,
            )
        end
    return _hybrid_beta_ula(
        data,
        prior;
        n_samples = n_samples,
        warmup = warmup,
        thin = thin,
        rng = rng,
        initial = init_vec,
        mala_step_size = mala_step_size,
        initial_step = initial_step,
        constraint = resolved_constraint,
        prox_sum0_lambda = prox_sum0_lambda,
        car_weights = car_weights,
        use_car = use_car,
        car_precision = car_precision,
        sample_car_strength = sample_car_strength,
        car_strength_prior_a = car_strength_prior_a,
        car_strength_prior_b = car_strength_prior_b,
        car_degree_scale = car_degree_scale,
        car_kappa_init = car_kappa_init,
        progress = progress,
        progress_desc = progress_desc,
        progress_offset = progress_offset,
    )
end

function _sample_ssl_prox_ula(
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior;
    n_samples::Integer,
    warmup::Integer,
    thin::Integer,
    rng::AbstractRNG,
    initial,
    mala_step_size::Real,
    constraint::Symbol,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    alpha_prior_kind = :invgamma,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
    init::Symbol = :zero,
    progress::Bool,
    progress_desc::AbstractString = "SSLASSO prox ULA",
    progress_offset::Integer = 0,
)
    p = size(data.z, 2)
    u = if initial !== nothing
        collect(Float64, initial)
    elseif init == :rand
        vcat(_default_unconstrained_init(rng, p), [0.0])  # η=0 → α=1 avoids gradient explosion with small prox_l1_lambda
    else
        vcat(zeros(Float64, p), [0.0])
    end
    length(u) == p + 1 || throw(DimensionMismatch("initial length must be p + 1 for prox SSLASSO ULA."))
    gamma = trues(p)
    theta = 0.5
    weights = _ssl_weights(prior, gamma)
    spike_weight = inv(prior.spike_scale)
    slab_weight = inv(prior.slab_scale)
    log_weight_odds = log(slab_weight) - log(spike_weight)
    names = vcat(["beta[$j]" for j in 1:p], ["eta"], ["gamma[$j]" for j in 1:p], ["theta"])
    d = length(names)
    draws = Matrix{Float64}(undef, n_samples, d)
    lps = Vector{Float64}(undef, n_samples)
    gamma_L = _check_positive(mala_step_size, "mala_step_size")
    total_iter = warmup + n_samples * thin
    current_lp = _ssl_prox_logposterior(
        u,
        data,
        prior,
        gamma,
        theta;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        alpha_prior_kind = alpha_prior_kind,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    saved = 0
    mala_work = MALAWorkspace(p + 1)
    prox_work = ProxGradientWorkspace(data)
    gradient_current! = (grad, uu) -> _ssl_prox_gradient!(
        grad,
        uu,
        data,
        prior,
        gamma,
        prox_work;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        alpha_prior_kind = alpha_prior_kind,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    prog = _progress_meter(progress, total_iter, "SSLASSO prox ULA"; progress_desc = progress_desc, progress_offset = progress_offset)
    for iter in 1:total_iter
        prop = _ula_step!(mala_work.proposal, gradient_current!, u, gamma_L, rng, mala_work)
        prop_lp = _ssl_prox_logposterior(
            prop,
            data,
            prior,
            gamma,
            theta;
            constraint = constraint,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = alpha_prior_shape,
            alpha_prior_scale = alpha_prior_scale,
            alpha_prior_kind = alpha_prior_kind,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
        )
        if isfinite(prop_lp)
            copyto!(u, prop)
            current_lp = prop_lp
        end

        beta = view(u, 1:p)
        alpha = exp(clamp(u[p + 1], -700.0, 700.0))
        theta_clamped = clamp(theta, eps(Float64), 1 - eps(Float64))
        for j in 1:p
            gamma[j] = false
            weights[j] = spike_weight
            h0 = weighted_l1_moreau(beta, alpha, weights, prox_l1_lambda;
                prox_root_tol = prox_root_tol, prox_root_maxiter = prox_root_maxiter)
            gamma[j] = true
            weights[j] = slab_weight
            h1 = weighted_l1_moreau(beta, alpha, weights, prox_l1_lambda;
                prox_root_tol = prox_root_tol, prox_root_maxiter = prox_root_maxiter)
            logit_p = log(theta_clamped) - log1p(-theta_clamped) + log_weight_odds - (h1 - h0)
            gamma[j] = rand(rng) < _logistic(logit_p)
            weights[j] = gamma[j] ? slab_weight : spike_weight
        end
        theta = rand(rng, Beta(prior.theta_a + count(gamma), prior.theta_b + p - count(gamma)))
        current_lp = _ssl_prox_logposterior(
            u,
            data,
            prior,
            gamma,
            theta;
            constraint = constraint,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = alpha_prior_shape,
            alpha_prior_scale = alpha_prior_scale,
            alpha_prior_kind = alpha_prior_kind,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
        )

        if iter > warmup && (iter - warmup) % thin == 0
            saved += 1
            @views draws[saved, 1:p] .= beta
            draws[saved, p + 1] = u[p + 1]
            @views draws[saved, (p + 2):(2p + 1)] .= Float64.(gamma)
            draws[saved, 2p + 2] = theta
            lps[saved] = current_lp
        end
        _progress_next!(prog)
    end
    _progress_finish!(prog)
    return MHSamples(draws, lps, names, NaN, NaN, fill(gamma_L, d))
end
