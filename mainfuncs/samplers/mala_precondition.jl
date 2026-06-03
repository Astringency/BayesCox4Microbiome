mutable struct PreconditionedMALAWorkspace
    grad_u::Vector{Float64}
    grad_prop::Vector{Float64}
    proposal::Vector{Float64}
    diff::Vector{Float64}
    noise::Vector{Float64}
    G_u::Matrix{Float64}
    G_prop::Matrix{Float64}
end

function PreconditionedMALAWorkspace(d::Integer)
    d > 0 || throw(ArgumentError("workspace dimension must be positive."))
    return PreconditionedMALAWorkspace(
        Vector{Float64}(undef, d),
        Vector{Float64}(undef, d),
        Vector{Float64}(undef, d),
        Vector{Float64}(undef, d),
        Vector{Float64}(undef, d),
        Matrix{Float64}(undef, d, d),
        Matrix{Float64}(undef, d, d),
    )
end

function _mala_sgld_epsilon(
    c0::Real,
    iter::Integer,
    n_obs::Integer,
    d_block::Integer,
    min_step::Real,
    max_step::Real,
)
    c0f = _check_positive(c0, "mala_sgld_c0")
    iter > 0 || throw(ArgumentError("iter must be positive."))
    n_obs > 0 || throw(ArgumentError("n_obs must be positive."))
    d_block > 0 || throw(ArgumentError("d_block must be positive."))
    lo = _check_positive(min_step, "mala_min_step")
    hi = _check_positive(max_step, "mala_max_step")
    hi >= lo || throw(ArgumentError("mala_max_step must be at least mala_min_step."))
    return clamp(c0f * Float64(iter)^(-0.55) / (Float64(n_obs) * Float64(d_block)), lo, hi)
end

function _validate_mala_schedule_options(
    mala_stepsize_schedule::Symbol,
    mala_sgld_c0::Real,
    mala_sgld_target_accept::Real,
    mala_sgld_accept_low::Real,
    mala_sgld_accept_high::Real,
    mala_sgld_adapt_every::Integer,
    mala_sgld_c0_min::Real,
    mala_sgld_c0_max::Real,
)
    mala_stepsize_schedule in (:adaptive_mcmc, :adaptive_sgld) ||
        throw(ArgumentError("mala_stepsize_schedule must be :adaptive_mcmc or :adaptive_sgld."))
    c0 = _check_positive(mala_sgld_c0, "mala_sgld_c0")
    c0_min = _check_positive(mala_sgld_c0_min, "mala_sgld_c0_min")
    c0_max = _check_positive(mala_sgld_c0_max, "mala_sgld_c0_max")
    c0_max > c0_min || throw(ArgumentError("mala_sgld_c0_max must be greater than mala_sgld_c0_min."))
    0 < mala_sgld_target_accept < 1 ||
        throw(ArgumentError("mala_sgld_target_accept must be in (0, 1)."))
    0 < mala_sgld_accept_low < mala_sgld_accept_high < 1 ||
        throw(ArgumentError("require 0 < mala_sgld_accept_low < mala_sgld_accept_high < 1."))
    mala_sgld_adapt_every > 0 ||
        throw(ArgumentError("mala_sgld_adapt_every must be positive."))
    return clamp(c0, c0_min, c0_max)
end

function _adapt_mala_sgld_c0(
    c0::Real,
    window_rate::Real,
    accept_low::Real,
    accept_high::Real,
    c0_min::Real,
    c0_max::Real,
)
    c0_new = Float64(c0)
    if window_rate > accept_high
        c0_new *= 1.2
    elseif window_rate < accept_low
        c0_new /= 1.5
    end
    return clamp(c0_new, Float64(c0_min), Float64(c0_max))
end

function _symmetrize_metric!(G::AbstractMatrix{Float64})
    size(G, 1) == size(G, 2) || throw(DimensionMismatch("metric must be square."))
    d = size(G, 1)
    @inbounds for j in 1:d
        if !isfinite(G[j, j])
            G[j, j] = 0.0
        end
        for i in (j + 1):d
            gij = isfinite(G[i, j]) ? G[i, j] : 0.0
            gji = isfinite(G[j, i]) ? G[j, i] : 0.0
            v = 0.5 * (gij + gji)
            G[i, j] = v
            G[j, i] = v
        end
    end
    return G
end

function _safe_metric_cholesky!(
    G::AbstractMatrix{Float64},
    ridge::Real,
    max_cholesky_tries::Integer,
)
    max_cholesky_tries > 0 || throw(ArgumentError("max_cholesky_tries must be positive."))
    d = size(G, 1)
    _symmetrize_metric!(G)
    base_diag = Vector{Float64}(undef, d)
    @inbounds for j in 1:d
        base_diag[j] = isfinite(G[j, j]) ? G[j, j] : 0.0
    end
    jitter = max(Float64(ridge), eps(Float64))
    for _ in 1:max_cholesky_tries
        @inbounds for j in 1:d
            G[j, j] = base_diag[j] + jitter
        end
        F = cholesky(Symmetric(G); check = false)
        if issuccess(F)
            return F
        end
        jitter *= 10.0
    end

    fill!(G, 0.0)
    @inbounds for j in 1:d
        v = base_diag[j]
        G[j, j] = isfinite(v) && v > 0.0 ? max(v, Float64(ridge), eps(Float64)) :
            max(Float64(ridge), eps(Float64))
    end
    return cholesky(Symmetric(G))
end

function _precond_mala_logq!(
    diff::AbstractVector{Float64},
    y::AbstractVector,
    x::AbstractVector,
    grad_x::AbstractVector{Float64},
    G_x::AbstractMatrix{Float64},
    F_x::Cholesky,
    epsilon::Real,
)
    epsf = Float64(epsilon)
    d = length(x)
    length(y) == d && length(grad_x) == d && length(diff) == d ||
        throw(DimensionMismatch("preconditioned MALA logq dimensions must match."))
    copyto!(diff, grad_x)
    ldiv!(F_x, diff)
    @inbounds for j in 1:d
        diff[j] = Float64(y[j]) - Float64(x[j]) - 0.5 * epsf * diff[j]
    end
    quad = 0.0
    @inbounds for i in 1:d
        row_sum = 0.0
        for j in 1:d
            row_sum += G_x[i, j] * diff[j]
        end
        quad += diff[i] * row_sum
    end
    logdetG = 2.0 * sum(log, diag(F_x.U))
    return 0.5 * logdetG - 0.5 * d * log(2pi * epsf) - 0.5 * quad / epsf
end

function _precond_mala_step!(
    logpost::Function,
    gradient!::Function,
    metric!::Function,
    u::Vector{Float64},
    current_lp::Float64,
    epsilon::Float64,
    rng::AbstractRNG,
    work::PreconditionedMALAWorkspace;
    ridge::Float64 = 1e-6,
    max_cholesky_tries::Int = 8,
)
    d = length(u)
    length(work.proposal) == d || throw(DimensionMismatch("preconditioned MALA workspace dimension mismatch."))
    epsilon > 0 || throw(ArgumentError("epsilon must be positive."))

    gradient!(work.grad_u, u)
    all(isfinite, work.grad_u) || return current_lp, false
    metric!(work.G_u, u)
    F_u = _safe_metric_cholesky!(work.G_u, ridge, max_cholesky_tries)

    copyto!(work.diff, work.grad_u)
    ldiv!(F_u, work.diff)
    randn!(rng, work.noise)
    ldiv!(F_u.U, work.noise)
    sqrt_eps = sqrt(epsilon)
    @inbounds for j in 1:d
        work.proposal[j] = u[j] + 0.5 * epsilon * work.diff[j] + sqrt_eps * work.noise[j]
    end

    prop_lp = logpost(work.proposal)
    isfinite(prop_lp) || return current_lp, false
    gradient!(work.grad_prop, work.proposal)
    all(isfinite, work.grad_prop) || return current_lp, false
    metric!(work.G_prop, work.proposal)
    F_prop = _safe_metric_cholesky!(work.G_prop, ridge, max_cholesky_tries)

    logq_forward = _precond_mala_logq!(
        work.diff,
        work.proposal,
        u,
        work.grad_u,
        work.G_u,
        F_u,
        epsilon,
    )
    logq_reverse = _precond_mala_logq!(
        work.diff,
        u,
        work.proposal,
        work.grad_prop,
        work.G_prop,
        F_prop,
        epsilon,
    )
    log_alpha = prop_lp + logq_reverse - current_lp - logq_forward
    accepted = isfinite(log_alpha) && log(rand(rng)) < log_alpha
    return accepted ? prop_lp : current_lp, accepted
end

function cox_observed_information_beta!(
    G::AbstractMatrix{Float64},
    data::CoxPreparedData,
    beta::AbstractVector,
    work = nothing;
    tie_method::Symbol = data.tie_method,
)
    tie = _check_cox_tie_method(tie_method)
    n, p = size(data.z)
    size(G) == (p, p) || throw(DimensionMismatch("G must be p by p."))
    length(beta) == p || throw(DimensionMismatch("length(beta) must match p."))
    fill!(G, 0.0)

    eta = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        s = 0.0
        for j in 1:p
            s += data.z[i, j] * Float64(beta[j])
        end
        eta[i] = s
    end
    all(isfinite, eta) || (fill!(G, NaN); return G)

    s1 = zeros(Float64, p)
    s2 = zeros(Float64, p, p)
    e1 = zeros(Float64, p)
    e2 = zeros(Float64, p, p)
    num1 = zeros(Float64, p)
    num2 = zeros(Float64, p, p)
    s0 = 0.0
    max_eta = -Inf

    @inbounds for k in eachindex(data.risk.group_start)
        for pos in data.risk.group_start[k]:data.risk.group_stop[k]
            i = data.risk.order[pos]
            if eta[i] > max_eta
                scale = isfinite(max_eta) ? exp(max_eta - eta[i]) : 0.0
                s0 *= scale
                s1 .*= scale
                s2 .*= scale
                max_eta = eta[i]
            end
            w = exp(eta[i] - max_eta)
            s0 += w
            for a in 1:p
                za = data.z[i, a]
                s1[a] += w * za
                for b in 1:p
                    s2[a, b] += w * za * data.z[i, b]
                end
            end
        end

        if tie == :breslow
            d = 0
            for pos in data.risk.group_start[k]:data.risk.group_stop[k]
                d += data.event[data.risk.order[pos]] != 0 ? 1 : 0
            end
            if d > 0
                for a in 1:p
                    ma = s1[a] / s0
                    for b in 1:p
                        G[a, b] += d * (s2[a, b] / s0 - ma * s1[b] / s0)
                    end
                end
            end
        else
            d = 0
            e0 = 0.0
            fill!(e1, 0.0)
            fill!(e2, 0.0)
            for pos in data.risk.group_start[k]:data.risk.group_stop[k]
                i = data.risk.order[pos]
                if data.event[i] != 0
                    d += 1
                    w = exp(eta[i] - max_eta)
                    e0 += w
                    for a in 1:p
                        za = data.z[i, a]
                        e1[a] += w * za
                        for b in 1:p
                            e2[a, b] += w * za * data.z[i, b]
                        end
                    end
                end
            end
            if d > 0
                for r in 0:(d - 1)
                    frac = r / d
                    denom = s0 - frac * e0
                    if !(denom > 0.0) || !isfinite(denom)
                        fill!(G, NaN)
                        return G
                    end
                    for a in 1:p
                        num1[a] = s1[a] - frac * e1[a]
                        for b in 1:p
                            num2[a, b] = s2[a, b] - frac * e2[a, b]
                        end
                    end
                    for a in 1:p
                        ma = num1[a] / denom
                        for b in 1:p
                            G[a, b] += num2[a, b] / denom - ma * num1[b] / denom
                        end
                    end
                end
            end
        end
    end
    return _symmetrize_metric!(G)
end

function constraint_observed_information_beta!(
    G::AbstractMatrix{Float64},
    beta::AbstractVector;
    constraint::Symbol,
    prox_sum0_lambda::Real,
)
    p = length(beta)
    size(G) == (p, p) || throw(DimensionMismatch("G must be p by p."))
    resolved = _resolve_constraint(constraint)
    if resolved == :soft_center
        v = 1.0 / _default_soft_center_scale(p)^2
        G .+= v
    elseif resolved == :prox
        lambda_sum0 = _check_positive(prox_sum0_lambda, "prox_sum0_lambda")
        G .+= 1.0 / (lambda_sum0 * p)
    end
    return G
end

function prior_observed_information_beta!(
    G::AbstractMatrix{Float64},
    u::AbstractVector,
    data::CoxPreparedData,
    prior::AbstractCoxPrior;
    ridge::Real = 1e-8,
)
    p = size(data.z, 2)
    size(G) == (p, p) || throw(DimensionMismatch("G must be p by p."))
    beta = view(u, 1:p)
    if prior isa HorseshoePrior
        log_tau = Float64(u[p + 1])
        log_lambda = view(u, (p + 2):(2p + 1))
        @inbounds for j in 1:p
            G[j, j] += exp(-2.0 * (log_tau + Float64(log_lambda[j])))
        end
    elseif prior isa RegularizedHorseshoePrior
        log_tau = Float64(u[p + 1])
        log_lambda = view(u, (p + 2):(2p + 1))
        c2 = exp(Float64(u[2p + 2]))
        tau = exp(log_tau)
        @inbounds for j in 1:p
            log_scale = _regularized_horseshoe_beta_logscale(tau, exp(Float64(log_lambda[j])), c2)
            G[j, j] += exp(-2.0 * log_scale)
        end
    elseif prior isa BayesianLassoPrior
        log_tau2 = view(u, (p + 1):(2p))
        @inbounds for j in 1:p
            G[j, j] += exp(-Float64(log_tau2[j]))
        end
    elseif prior isa SpikeSlabLassoPrior
        log_tau2_spike = view(u, (p + 1):(2p))
        log_tau2_slab = view(u, (2p + 1):(3p))
        theta = clamp(_logistic(Float64(u[3p + 1])), eps(Float64), 1 - eps(Float64))
        r = Float64(ridge)
        @inbounds for j in 1:p
            b = Float64(beta[j])
            tau2_spike = exp(Float64(log_tau2_spike[j]))
            tau2_slab = exp(Float64(log_tau2_slab[j]))
            log_spike, log_slab, norm =
                _ssl_component_logpdf(b, theta, log_tau2_spike[j], log_tau2_slab[j])
            prob_spike = exp(log_spike - norm)
            prob_slab = exp(log_slab - norm)
            A = prob_spike / tau2_spike + prob_slab / tau2_slab
            B = prob_spike / tau2_spike^2 + prob_slab / tau2_slab^2
            Gjj = A - b^2 * (B - A^2)
            if !isfinite(Gjj)
                Gjj = A
            elseif Gjj <= 0.0
                Gjj = max(A, r)
            end
            G[j, j] += Gjj
        end
    else
        throw(ArgumentError("unsupported prior type for beta observed information: $(typeof(prior))"))
    end
    return G
end

function _add_ridge!(G::AbstractMatrix{Float64}, ridge::Real)
    r = Float64(ridge)
    r == 0.0 && return G
    @inbounds for j in 1:min(size(G, 1), size(G, 2))
        G[j, j] += r
    end
    return G
end

function _diag_only!(G::AbstractMatrix{Float64})
    d1, d2 = size(G)
    @inbounds for j in 1:d2, i in 1:d1
        i == j || (G[i, j] = 0.0)
    end
    return G
end

function standard_observed_information_beta!(
    G::AbstractMatrix{Float64},
    data::CoxPreparedData,
    prior::AbstractCoxPrior,
    u::AbstractVector;
    constraint::Symbol,
    prox_sum0_lambda::Real,
    ridge::Real = 0.0,
    diag_only::Bool = false,
)
    p = size(data.z, 2)
    length(u) >= p || throw(DimensionMismatch("state is shorter than beta dimension."))
    cox_observed_information_beta!(G, data, view(u, 1:p))
    constraint_observed_information_beta!(
        G,
        view(u, 1:p);
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    prior_observed_information_beta!(G, u, data, prior; ridge = max(Float64(ridge), eps(Float64)))
    diag_only && _diag_only!(G)
    _add_ridge!(G, ridge)
    return _symmetrize_metric!(G)
end

function _lasso_moreau_observed_information!(
    G::AbstractMatrix{Float64},
    beta::AbstractVector,
    eta::Real,
    lambda_l1::Real,
    alpha_prior_scale::Real;
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = length(beta)
    lambda = _check_positive(lambda_l1, "prox_l1_lambda")
    alpha = exp(clamp(Float64(eta), -700.0, 700.0))
    if sum(abs, beta) > alpha
        nu = _l1_projection_root(
            beta,
            alpha;
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
        )
        active = [j for j in 1:p if abs(Float64(beta[j])) > nu]
        isactive = falses(p)
        for j in active
            isactive[j] = true
        end
        m = length(active)
        denom = lambda * (m + 1)
        for a in active, b in active
            G[a, b] += sign(Float64(beta[a])) * sign(Float64(beta[b])) / denom
        end
        for j in 1:p
            if !isactive[j]
                G[j, j] += 1.0 / lambda
            end
        end
        eta_idx = p + 1
        for j in active
            v = -alpha * sign(Float64(beta[j])) / denom
            G[j, eta_idx] += v
            G[eta_idx, j] += v
        end
        G[eta_idx, eta_idx] += (-alpha * nu + alpha^2 / (m + 1)) / lambda
    end
    G[p + 1, p + 1] += Float64(alpha_prior_scale) / alpha
    return G
end

function _weighted_lasso_moreau_observed_information!(
    G::AbstractMatrix{Float64},
    beta::AbstractVector,
    eta::Real,
    weights::AbstractVector,
    lambda_l1::Real,
    alpha_prior_scale::Real;
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = length(beta)
    lambda = _check_positive(lambda_l1, "prox_l1_lambda")
    alpha = exp(clamp(Float64(eta), -700.0, 700.0))
    if sum(Float64(weights[j]) * abs(Float64(beta[j])) for j in 1:p) > alpha
        nu = _l1_projection_root(
            beta,
            alpha;
            weights = weights,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
        )
        active = [j for j in 1:p if abs(Float64(beta[j])) > nu * Float64(weights[j])]
        isactive = falses(p)
        for j in active
            isactive[j] = true
        end
        D = 1.0 + sum(Float64(weights[j])^2 for j in active)
        denom = lambda * D
        for a in active, b in active
            ua = Float64(weights[a]) * sign(Float64(beta[a]))
            ub = Float64(weights[b]) * sign(Float64(beta[b]))
            G[a, b] += ua * ub / denom
        end
        for j in 1:p
            if !isactive[j]
                G[j, j] += 1.0 / lambda
            end
        end
        eta_idx = p + 1
        for j in active
            uaj = Float64(weights[j]) * sign(Float64(beta[j]))
            v = -alpha * uaj / denom
            G[j, eta_idx] += v
            G[eta_idx, j] += v
        end
        G[eta_idx, eta_idx] += (-alpha * nu + alpha^2 / D) / lambda
    end
    G[p + 1, p + 1] += Float64(alpha_prior_scale) / alpha
    return G
end

function prox_lasso_observed_information!(
    G::AbstractMatrix{Float64},
    u::AbstractVector,
    data::CoxPreparedData,
    prior::BayesianLassoPrior;
    constraint::Symbol,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
    ridge::Real = 0.0,
    diag_only::Bool = false,
)
    p = size(data.z, 2)
    length(u) == p + 1 || throw(DimensionMismatch("u length must be p + 1."))
    size(G) == (p + 1, p + 1) || throw(DimensionMismatch("G must be (p + 1) by (p + 1)."))
    fill!(G, 0.0)
    beta = view(u, 1:p)
    cox_observed_information_beta!(view(G, 1:p, 1:p), data, beta)
    constraint_observed_information_beta!(
        view(G, 1:p, 1:p),
        beta;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    _lasso_moreau_observed_information!(
        G,
        beta,
        u[p + 1],
        prox_l1_lambda,
        alpha_prior_scale;
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    diag_only && _diag_only!(G)
    _add_ridge!(G, ridge)
    return _symmetrize_metric!(G)
end

function prox_ssl_observed_information!(
    G::AbstractMatrix{Float64},
    u::AbstractVector,
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior,
    gamma::AbstractVector{Bool};
    constraint::Symbol,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
    ridge::Real = 0.0,
    diag_only::Bool = false,
)
    p = size(data.z, 2)
    length(gamma) == p || throw(DimensionMismatch("gamma length must match p."))
    length(u) == p + 1 || throw(DimensionMismatch("u length must be p + 1."))
    size(G) == (p + 1, p + 1) || throw(DimensionMismatch("G must be (p + 1) by (p + 1)."))
    fill!(G, 0.0)
    beta = view(u, 1:p)
    cox_observed_information_beta!(view(G, 1:p, 1:p), data, beta)
    constraint_observed_information_beta!(
        view(G, 1:p, 1:p),
        beta;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    weights = _ssl_weights(prior, gamma)
    _weighted_lasso_moreau_observed_information!(
        G,
        beta,
        u[p + 1],
        weights,
        prox_l1_lambda,
        alpha_prior_scale;
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    diag_only && _diag_only!(G)
    _add_ridge!(G, ridge)
    return _symmetrize_metric!(G)
end

function _defined_type(sym::Symbol)
    return isdefined(@__MODULE__, sym) ? getfield(@__MODULE__, sym) : Union{}
end

function observed_information!(
    G::AbstractMatrix{Float64},
    target,
    u::AbstractVector;
    ridge::Real,
    mode::Symbol = :auto,
    diag_only::Bool = false,
)
    mode in (:auto, :beta_dense, :beta_diag, :full_dense) ||
        throw(ArgumentError("mode must be :auto, :beta_dense, :beta_diag, or :full_dense."))
    posterior_type = _defined_type(:CoxPosteriorTarget)
    prox_lasso_type = _defined_type(:CoxProxLassoTarget)
    prox_ssl_type = _defined_type(:CoxProxSSLassoTarget)
    noncentered_type = _defined_type(:CoxNoncenteredPosteriorTarget)

    if target isa posterior_type
        actual_mode = mode == :auto ? :beta_dense : mode
        actual_mode in (:beta_dense, :beta_diag) ||
            throw(ArgumentError("standard Cox targets support beta-block observed information only."))
        return standard_observed_information_beta!(
            G,
            target.data,
            target.prior,
            u;
            constraint = target.constraint,
            prox_sum0_lambda = target.prox_sum0_lambda,
            ridge = ridge,
            diag_only = diag_only || actual_mode == :beta_diag,
        )
    elseif target isa prox_lasso_type
        actual_mode = mode == :auto ? :full_dense : mode
        actual_mode == :full_dense ||
            throw(ArgumentError("prox-lasso observed information uses the full continuous block."))
        return prox_lasso_observed_information!(
            G,
            u,
            target.data,
            target.prior;
            constraint = target.constraint,
            prox_sum0_lambda = target.prox_sum0_lambda,
            prox_l1_lambda = target.prox_l1_lambda,
            alpha_prior_shape = target.alpha_prior_shape,
            alpha_prior_scale = target.alpha_prior_scale,
            prox_root_tol = target.prox_root_tol,
            prox_root_maxiter = target.prox_root_maxiter,
            ridge = ridge,
            diag_only = diag_only,
        )
    elseif target isa prox_ssl_type
        actual_mode = mode == :auto ? :full_dense : mode
        actual_mode == :full_dense ||
            throw(ArgumentError("prox-sslasso observed information uses the full continuous block."))
        return prox_ssl_observed_information!(
            G,
            u,
            target.data,
            target.prior,
            target.gamma;
            constraint = target.constraint,
            prox_sum0_lambda = target.prox_sum0_lambda,
            prox_l1_lambda = target.prox_l1_lambda,
            alpha_prior_shape = target.alpha_prior_shape,
            alpha_prior_scale = target.alpha_prior_scale,
            prox_root_tol = target.prox_root_tol,
            prox_root_maxiter = target.prox_root_maxiter,
            ridge = ridge,
            diag_only = diag_only,
        )
    elseif target isa noncentered_type
        throw(ArgumentError("observed_information! is not implemented for noncentered Cox targets."))
    end
    throw(ArgumentError("unsupported Cox target type for observed_information!: $(typeof(target))"))
end
