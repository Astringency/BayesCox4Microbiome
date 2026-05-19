function _ssl_normal_prior_logpdf(beta::AbstractVector, tau2::AbstractVector)
    lp = zero(eltype(beta))
    @inbounds for j in eachindex(beta)
        lp += _normal_zero_logpdf_from_logscale(beta[j], 0.5 * log(tau2[j]))
    end
    return lp
end

function _ssl_joint_logposterior(
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior,
    beta::AbstractVector,
    gamma::AbstractVector{Bool},
    tau2::AbstractVector,
    theta::Real;
    constraint::Symbol,
    soft_center_scale::Float64,
    prox_sum0_lambda::Real,
)
    0 < theta < 1 || return -Inf
    p = length(beta)
    lp = cox_logpartiallikelihood(data, beta)
    lp += _constraint_logpenalty(
        beta;
        constraint = constraint,
        soft_center_scale = soft_center_scale,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    lp += logpdf(Beta(prior.theta_a, prior.theta_b), theta)
    @inbounds for j in 1:p
        scale = gamma[j] ? prior.slab_scale : prior.spike_scale
        lp += (gamma[j] ? log(theta) : log1p(-theta))
        lp += _ssl_tau2_logpdf(tau2[j], scale)
        lp += _normal_zero_logpdf_from_logscale(beta[j], 0.5 * log(tau2[j]))
    end
    return lp
end

function _ssl_beta_logtarget(
    data::CoxPreparedData,
    beta::AbstractVector,
    tau2::AbstractVector;
    constraint::Symbol,
    soft_center_scale::Float64,
    prox_sum0_lambda::Real,
)
    lp = cox_logpartiallikelihood(data, beta) + _ssl_normal_prior_logpdf(beta, tau2)
    lp += _constraint_logpenalty(
        beta;
        constraint = constraint,
        soft_center_scale = soft_center_scale,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    return lp
end

function _sample_ssl_tau2(rng::AbstractRNG, beta::Real, scale::Real)
    lambda2 = inv(scale^2)
    absbeta = abs(beta)
    if absbeta <= sqrt(eps(Float64))
        return rand(rng, Gamma(0.5, 2 * scale^2))
    end
    invtau2 = rand(rng, InverseGaussian(inv(scale * absbeta), lambda2))
    return inv(invtau2)
end

function _sample_ssl_gibbs(
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior;
    n_samples::Integer = 5000,
    warmup::Integer = 2000,
    thin::Integer = 1,
    rng::AbstractRNG = Random.default_rng(),
    initial_beta = nothing,
    initial_step::Union{Real,AbstractVector{<:Real}} = 0.05,
    target_accept::Real = 0.234,
    adapt_every::Integer = 50,
    min_step::Real = 1e-5,
    max_step::Real = 1.0,
    constraint::Symbol = :soft_center,
    soft_center = nothing,
    soft_center_scale::Union{Nothing,Real} = nothing,
    prox_sum0_lambda::Real = 1e-3,
    center_beta_proposal::Bool = true,
    chains::Integer = 1,
    progress::Bool = true,
    progress_desc::AbstractString = "Gibbs sampling",
    progress_offset::Integer = 0,
)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive."))
    warmup >= 0 || throw(ArgumentError("warmup must be nonnegative."))
    thin > 0 || throw(ArgumentError("thin must be positive."))
    chains == 1 || throw(ArgumentError("SSLASSO Gibbs currently supports chains=1. Run independent seeds for multiple chains."))
    progress isa Bool || throw(ArgumentError("progress must be Bool."))
    adapt_every > 0 || throw(ArgumentError("adapt_every must be positive."))
    _validate_prior(prior)

    p = size(data.z, 2)
    beta = initial_beta === nothing ? zeros(Float64, p) : collect(Float64, initial_beta)
    length(beta) == p || throw(DimensionMismatch("initial_beta length must match p."))
    theta = 0.5
    gamma = trues(p)
    tau2 = [_sample_ssl_tau2(rng, beta[j], prior.slab_scale) for j in 1:p]
    resolved_constraint = _resolve_constraint(constraint, soft_center)
    sc_scale = soft_center_scale === nothing ? _default_soft_center_scale(p) : Float64(soft_center_scale)

    d = p
    step = initial_step isa Real ? fill(Float64(initial_step), d) : collect(Float64, initial_step)
    length(step) == d || throw(DimensionMismatch("initial_step length must match beta dimension."))
    all(>(0), step) || throw(ArgumentError("initial_step entries must be positive."))

    names = vcat(
        ["beta[$j]" for j in 1:p],
        ["gamma[$j]" for j in 1:p],
        ["tau2[$j]" for j in 1:p],
        ["theta"],
    )
    draws = Matrix{Float64}(undef, n_samples, length(names))
    lps = Vector{Float64}(undef, n_samples)
    total_iter = warmup + n_samples * thin

    current_beta_lp = _ssl_beta_logtarget(
        data,
        beta,
        tau2;
        constraint = resolved_constraint,
        soft_center_scale = sc_scale,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    isfinite(current_beta_lp) || throw(ArgumentError("initial beta has non-finite log posterior."))

    mu = copy(beta)
    m2 = zeros(p)
    n_adapt = 1
    global_scale = 1.0
    accept_warmup = 0
    accept_sample = 0
    accept_window = 0
    saved = 0
    prog = _progress_meter(progress, total_iter, "Gibbs sampling"; progress_desc = progress_desc, progress_offset = progress_offset)

    for iter in 1:total_iter
        noise = step .* randn(rng, p)
        if center_beta_proposal && p > 1
            noise .-= mean(noise)
            if resolved_constraint == :soft_center
                noise .+= sc_scale * randn(rng) / p
            end
        end
        beta_prop = beta .+ global_scale .* noise
        prop_lp = _ssl_beta_logtarget(
            data,
            beta_prop,
            tau2;
            constraint = resolved_constraint,
            soft_center_scale = sc_scale,
            prox_sum0_lambda = prox_sum0_lambda,
        )
        accepted = isfinite(prop_lp) && log(rand(rng)) < prop_lp - current_beta_lp
        if accepted
            beta = beta_prop
            current_beta_lp = prop_lp
            if iter <= warmup
                accept_warmup += 1
                accept_window += 1
            else
                accept_sample += 1
            end
        end

        @inbounds for j in 1:p
            log_slab = log(theta) + _ssl_tau2_logpdf(tau2[j], prior.slab_scale)
            log_spike = log1p(-theta) + _ssl_tau2_logpdf(tau2[j], prior.spike_scale)
            prob_slab = exp(log_slab - _logsumexp2(log_spike, log_slab))
            gamma[j] = rand(rng) < prob_slab
        end
        theta = rand(rng, Beta(prior.theta_a + count(gamma), prior.theta_b + p - count(gamma)))
        @inbounds for j in 1:p
            scale = gamma[j] ? prior.slab_scale : prior.spike_scale
            tau2[j] = _sample_ssl_tau2(rng, beta[j], scale)
        end
        current_beta_lp = _ssl_beta_logtarget(
            data,
            beta,
            tau2;
            constraint = resolved_constraint,
            soft_center_scale = sc_scale,
            prox_sum0_lambda = prox_sum0_lambda,
        )

        if iter <= warmup
            n_adapt += 1
            delta = beta .- mu
            mu .+= delta ./ n_adapt
            m2 .+= delta .* (beta .- mu)
            if iter % adapt_every == 0 && n_adapt > 20
                empirical_sd = sqrt.(max.(m2 ./ max(n_adapt - 1, 1), eps(Float64)))
                step .= clamp.(2.38 / sqrt(p) .* empirical_sd, min_step, max_step)
                window_rate = accept_window / adapt_every
                global_scale *= exp(clamp(window_rate - target_accept, -0.25, 0.25))
                accept_window = 0
            end
        elseif (iter - warmup) % thin == 0
            saved += 1
            @views draws[saved, 1:p] .= beta
            @views draws[saved, (p + 1):(2p)] .= Float64.(gamma)
            @views draws[saved, (2p + 1):(3p)] .= tau2
            draws[saved, 3p + 1] = theta
            lps[saved] = _ssl_joint_logposterior(
                data,
                prior,
                beta,
                gamma,
                tau2,
                theta;
                constraint = resolved_constraint,
                soft_center_scale = sc_scale,
                prox_sum0_lambda = prox_sum0_lambda,
            )
        end
        phase = iter <= warmup ? "warmup $(iter)/$(warmup)" : "sample $(saved)/$(n_samples)"
        _progress_next!(prog; showvalues = [(:phase, phase), (:saved, saved)])
    end
    _progress_finish!(prog)

    sample_iters = n_samples * thin
    warmup_rate = warmup == 0 ? NaN : accept_warmup / warmup
    sample_rate = sample_iters == 0 ? NaN : accept_sample / sample_iters
    final_step = global_scale .* step
    return MHSamples(draws, lps, names, sample_rate, warmup_rate, final_step)
end

function _sample_coordinate_gibbs(
    data::CoxPreparedData,
    prior::AbstractCoxPrior;
    n_samples::Integer = 5000,
    warmup::Integer = 2000,
    thin::Integer = 1,
    rng::AbstractRNG = Random.default_rng(),
    initial = nothing,
    initial_step::Union{Real,AbstractVector{<:Real}} = 0.05,
    target_accept::Real = 0.44,
    adapt_every::Integer = 50,
    min_step::Real = 1e-5,
    max_step::Real = 1.0,
    constraint::Symbol = :soft_center,
    soft_center = nothing,
    soft_center_scale::Union{Nothing,Real} = nothing,
    prox_sum0_lambda::Real = 1e-3,
    car_weights = nothing,
    use_car::Bool = car_weights !== nothing,
    car_precision::Real = 1.0,
    progress::Bool = true,
    progress_desc::AbstractString = "Coordinate Gibbs sampling",
    progress_offset::Integer = 0,
    kwargs...,
)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive."))
    warmup >= 0 || throw(ArgumentError("warmup must be nonnegative."))
    thin > 0 || throw(ArgumentError("thin must be positive."))
    adapt_every > 0 || throw(ArgumentError("adapt_every must be positive."))
    0 < target_accept < 1 || throw(ArgumentError("target_accept must be in (0, 1)."))
    _validate_prior(prior)

    p = size(data.z, 2)
    n_events = count(!=(0), data.event)
    u =
        if initial !== nothing
            collect(Float64, initial)
        elseif prior isa RegularizedHorseshoePrior
            initial_state(prior, p, n_events; n_obs = size(data.z, 1))
        else
            initial_state(prior, p, n_events)
        end
    names = _state_names(prior, p)
    d = length(u)
    length(names) == d || throw(DimensionMismatch("initial state length does not match prior state."))
    step = initial_step isa Real ? fill(Float64(initial_step), d) : collect(Float64, initial_step)
    length(step) == d || throw(DimensionMismatch("initial_step length must match state dimension."))
    all(>(0), step) || throw(ArgumentError("initial_step entries must be positive."))
    min_s = _check_positive(min_step, "min_step")
    max_s = _check_positive(max_step, "max_step")

    resolved_constraint = _resolve_constraint(constraint, soft_center)
    sc_scale = soft_center_scale === nothing ? _default_soft_center_scale(p) : Float64(soft_center_scale)
    logpost = uu -> logposterior(
        uu,
        data,
        prior;
        constraint = resolved_constraint,
        soft_center_scale = sc_scale,
        prox_sum0_lambda = prox_sum0_lambda,
        car_weights = car_weights,
        use_car = use_car,
        car_precision = car_precision,
    )

    total_iter = warmup + n_samples * thin
    draws = Matrix{Float64}(undef, n_samples, d)
    lps = Vector{Float64}(undef, n_samples)
    current_lp = logpost(u)
    isfinite(current_lp) || throw(ArgumentError("initial state has non-finite log posterior."))

    accept_window = zeros(Int, d)
    accept_warmup = zeros(Int, d)
    accept_sample = zeros(Int, d)
    saved = 0
    proposal = copy(u)
    prog = _progress_meter(progress, total_iter, "Coordinate Gibbs sampling"; progress_desc = progress_desc, progress_offset = progress_offset)

    for iter in 1:total_iter
        for j in 1:d
            copyto!(proposal, u)
            proposal[j] += step[j] * randn(rng)
            prop_lp = logpost(proposal)
            accepted = isfinite(prop_lp) && log(rand(rng)) < prop_lp - current_lp
            if accepted
                u[j] = proposal[j]
                current_lp = prop_lp
                if iter <= warmup
                    accept_warmup[j] += 1
                    accept_window[j] += 1
                else
                    accept_sample[j] += 1
                end
            end
        end

        if iter <= warmup && iter % adapt_every == 0
            rates = accept_window ./ adapt_every
            step .= clamp.(step .* exp.(clamp.(rates .- target_accept, -0.25, 0.25)), min_s, max_s)
            fill!(accept_window, 0)
        elseif iter > warmup && (iter - warmup) % thin == 0
            saved += 1
            @views draws[saved, :] .= u
            lps[saved] = current_lp
        end
        phase = iter <= warmup ? "warmup $(iter)/$(warmup)" : "sample $(saved)/$(n_samples)"
        _progress_next!(prog; showvalues = [(:phase, phase), (:saved, saved)])
    end
    _progress_finish!(prog)

    sample_iters = n_samples * thin
    warmup_rate = warmup == 0 ? NaN : mean(accept_warmup ./ warmup)
    sample_rate = sample_iters == 0 ? NaN : mean(accept_sample ./ sample_iters)
    return MHSamples(draws, lps, names, sample_rate, warmup_rate, step)
end
