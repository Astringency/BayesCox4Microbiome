function _advancedhmc_acceptance_rate(stats)
    isempty(stats) && return NaN
    vals = Float64[]
    for stat in stats
        if haskey(stat, :acceptance_rate)
            push!(vals, Float64(stat.acceptance_rate))
        end
    end
    return isempty(vals) ? NaN : mean(vals)
end

function _sample_explicit_nuts_one(
    target::AbstractCoxTarget;
    n_samples::Integer,
    warmup::Integer,
    rng::AbstractRNG,
    target_accept::Real,
    max_depth::Integer,
    progress::Bool,
    progress_desc::AbstractString = "NUTS sampling",
    progress_offset::Integer = 0,
    initial = nothing,
)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive."))
    warmup >= 0 || throw(ArgumentError("warmup must be nonnegative."))
    max_depth > 0 || throw(ArgumentError("max_depth must be positive."))
    0 < target_accept < 1 || throw(ArgumentError("target_accept must be in (0, 1)."))

    init = initial === nothing ? initial_state(target) : collect(Float64, initial)
    d = dimension(target)
    length(init) == d || throw(DimensionMismatch("initial length must match target dimension."))
    isfinite(logdensity(target, init)) ||
        throw(ArgumentError("initial state has non-finite log posterior."))

    logp = u -> logdensity(target, u)
    gradlogp = u -> LogDensityProblems.logdensity_and_gradient(target, u)
    metric = AdvancedHMC.DiagEuclideanMetric(d)
    hamiltonian = AdvancedHMC.Hamiltonian(metric, logp, gradlogp)
    initial_step = AdvancedHMC.find_good_stepsize(rng, hamiltonian, init)
    integrator = AdvancedHMC.Leapfrog(initial_step)
    kernel = AdvancedHMC.HMCKernel(
        AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(
            integrator,
            AdvancedHMC.GeneralisedNoUTurn(max_depth = Int(max_depth)),
        ),
    )
    adaptor = AdvancedHMC.StanHMCAdaptor(
        AdvancedHMC.MassMatrixAdaptor(metric),
        AdvancedHMC.StepSizeAdaptor(Float64(target_accept), integrator),
    )

    total_iter = warmup + n_samples
    progress_next! = (prog, stat) -> _progress_hmc_next!(
        prog,
        stat;
        progress_desc = progress_desc,
        progress_offset = progress_offset,
    )
    samples, stats = AdvancedHMC.sample(
        rng,
        hamiltonian,
        kernel,
        init,
        Int(total_iter),
        adaptor,
        Int(warmup);
        drop_warmup = warmup > 0,
        progress = progress,
        verbose = false,
        pm_next! = progress_next!,
    )

    draws = Matrix{Float64}(undef, n_samples, d)
    @inbounds for i in 1:n_samples
        draws[i, :] .= samples[i]
    end
    lps = [logdensity(target, view(draws, i, :)) for i in axes(draws, 1)]
    return MHSamples(
        draws,
        lps,
        state_names(target),
        _advancedhmc_acceptance_rate(stats),
        NaN,
        fill(NaN, d),
    )
end

struct CoxPosteriorTarget <: AbstractCoxTarget
    data::CoxPreparedData
    prior::AbstractCoxPrior
    constraint::Symbol
    soft_center_scale::Float64
    prox_sum0_lambda::Float64
    car_weights
    use_car::Bool
    car_precision::Float64
    names::Vector{String}
    work::CoxGradientWorkspace
end

dimension(target::CoxPosteriorTarget) = length(target.names)
state_names(target::CoxPosteriorTarget) = target.names

function CoxPosteriorTarget(
    data::CoxPreparedData,
    prior::AbstractCoxPrior;
    constraint::Symbol,
    soft_center_scale::Real,
    prox_sum0_lambda::Real,
    car_weights = nothing,
    use_car::Bool = car_weights !== nothing,
    car_precision::Real = 1.0,
)
    p = size(data.z, 2)
    return CoxPosteriorTarget(
        data,
        prior,
        constraint,
        Float64(soft_center_scale),
        Float64(prox_sum0_lambda),
        car_weights,
        use_car,
        Float64(car_precision),
        _state_names(prior, p),
        CoxGradientWorkspace(data),
    )
end

function logdensity(target::CoxPosteriorTarget, u::AbstractVector)
    return logposterior(
        u,
        target.data,
        target.prior;
        constraint = target.constraint,
        soft_center_scale = target.soft_center_scale,
        prox_sum0_lambda = target.prox_sum0_lambda,
        car_weights = target.car_weights,
        use_car = target.use_car,
        car_precision = target.car_precision,
    )
end

function gradient!(grad::AbstractVector{Float64}, target::CoxPosteriorTarget, u::AbstractVector)
    logposterior_and_gradient!(
        grad,
        u,
        target.data,
        target.prior,
        target.work;
        constraint = target.constraint,
        soft_center_scale = target.soft_center_scale,
        prox_sum0_lambda = target.prox_sum0_lambda,
        car_weights = target.car_weights,
        use_car = target.use_car,
        car_precision = target.car_precision,
    )
    return grad
end

function logdensity_and_gradient!(
    grad::AbstractVector{Float64},
    target::CoxPosteriorTarget,
    u::AbstractVector,
)
    return logposterior_and_gradient!(
        grad,
        u,
        target.data,
        target.prior,
        target.work;
        constraint = target.constraint,
        soft_center_scale = target.soft_center_scale,
        prox_sum0_lambda = target.prox_sum0_lambda,
        car_weights = target.car_weights,
        use_car = target.use_car,
        car_precision = target.car_precision,
    )
end

function _generic_explicit_initial(prior::AbstractCoxPrior, data::CoxPreparedData, initial)
    initial !== nothing && return collect(Float64, initial)
    p = size(data.z, 2)
    n_events = count(!=(0), data.event)
    if prior isa RegularizedHorseshoePrior
        return initial_state(prior, p, n_events; n_obs = size(data.z, 1))
    else
        return initial_state(prior, p, n_events)
    end
end

function _make_explicit_nuts_target(
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
    car_weights = nothing,
    use_car::Bool = car_weights !== nothing,
    car_precision::Real = 1.0,
    kwargs...,
)
    if constraint != :prox
        return CoxPosteriorTarget(
            data,
            prior;
            constraint = constraint,
            soft_center_scale = soft_center_scale,
            prox_sum0_lambda = prox_sum0_lambda,
            car_weights = car_weights,
            use_car = use_car,
            car_precision = car_precision,
        )
    end
    return CoxProxLassoTarget(
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
end

function _make_explicit_nuts_target(
    data::CoxPreparedData,
    prior::AbstractCoxPrior;
    constraint::Symbol,
    soft_center_scale::Real,
    prox_sum0_lambda::Real,
    car_weights = nothing,
    use_car::Bool = car_weights !== nothing,
    car_precision::Real = 1.0,
    kwargs...,
)
    return CoxPosteriorTarget(
        data,
        prior;
        constraint = constraint,
        soft_center_scale = soft_center_scale,
        prox_sum0_lambda = prox_sum0_lambda,
        car_weights = car_weights,
        use_car = use_car,
        car_precision = car_precision,
    )
end

function _sample_ssl_prox_explicit_nuts_gibbs(
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior;
    n_samples::Integer,
    warmup::Integer,
    thin::Integer,
    chains::Integer,
    target_accept::Real,
    max_depth::Integer,
    rng::AbstractRNG,
    constraint::Symbol,
    soft_center_scale::Real,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape,
    alpha_prior_rate,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
    progress::Bool,
    parallel_chains::Bool,
)
    timed = @timed begin
        if chains == 1
            _sample_ssl_prox_explicit_nuts_gibbs_one(
                data,
                prior;
                n_samples = n_samples,
                warmup = warmup,
                thin = thin,
                rng = rng,
                target_accept = target_accept,
                max_depth = max_depth,
                constraint = constraint,
                soft_center_scale = soft_center_scale,
                prox_sum0_lambda = prox_sum0_lambda,
                prox_l1_lambda = prox_l1_lambda,
                alpha_prior_shape = alpha_prior_shape,
                alpha_prior_rate = alpha_prior_rate,
                prox_root_tol = prox_root_tol,
                prox_root_maxiter = prox_root_maxiter,
                progress = progress,
                progress_desc = "SSLASSO explicit prox NUTS-Gibbs",
                progress_offset = 0,
            )
        else
            samples = Vector{MHSamples}(undef, chains)
            rngs = [_independent_rng(rng) for _ in 1:chains]
            if parallel_chains
                _reserve_chain_progress_lines(chains, progress)
                Threads.@threads for c in 1:chains
                    samples[c] = _sample_ssl_prox_explicit_nuts_gibbs_one(
                        data,
                        prior;
                        n_samples = n_samples,
                        warmup = warmup,
                        thin = thin,
                        rng = rngs[c],
                        target_accept = target_accept,
                        max_depth = max_depth,
                        constraint = constraint,
                        soft_center_scale = soft_center_scale,
                        prox_sum0_lambda = prox_sum0_lambda,
                        prox_l1_lambda = prox_l1_lambda,
                        alpha_prior_shape = alpha_prior_shape,
                        alpha_prior_rate = alpha_prior_rate,
                        prox_root_tol = prox_root_tol,
                        prox_root_maxiter = prox_root_maxiter,
                        progress = progress,
                        progress_desc = _chain_progress_desc("SSLASSO explicit prox NUTS-Gibbs", c, chains),
                        progress_offset = _chain_progress_offset(c, chains),
                    )
                end
            else
                for c in 1:chains
                    @info "Running explicit SSLASSO prox NUTS-Gibbs chain $c/$chains"
                    samples[c] = _sample_ssl_prox_explicit_nuts_gibbs_one(
                        data,
                        prior;
                        n_samples = n_samples,
                        warmup = warmup,
                        thin = thin,
                        rng = rngs[c],
                        target_accept = target_accept,
                        max_depth = max_depth,
                        constraint = constraint,
                        soft_center_scale = soft_center_scale,
                        prox_sum0_lambda = prox_sum0_lambda,
                        prox_l1_lambda = prox_l1_lambda,
                        alpha_prior_shape = alpha_prior_shape,
                        alpha_prior_rate = alpha_prior_rate,
                        prox_root_tol = prox_root_tol,
                        prox_root_maxiter = prox_root_maxiter,
                        progress = progress,
                    )
                end
            end
            MultiMHSamples(samples)
        end
    end
    gc_percent = timed.time > 0 ? 100 * timed.gctime / timed.time : NaN
    diagnostics = SamplingDiagnostics(missing, missing, timed.time, Int64(timed.bytes), timed.gctime, gc_percent)
    return (samples = timed.value, diagnostics = diagnostics)
end

function _sample_ssl_prox_explicit_nuts_gibbs_one(
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior;
    n_samples::Integer,
    warmup::Integer,
    thin::Integer,
    rng::AbstractRNG,
    target_accept::Real,
    max_depth::Integer,
    constraint::Symbol,
    soft_center_scale::Real,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape,
    alpha_prior_rate,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
    progress::Bool,
    progress_desc::AbstractString = "SSLASSO explicit prox NUTS-Gibbs",
    progress_offset::Integer = 0,
)
    p = size(data.z, 2)
    a_shape, a_rate = _prox_alpha_hyperparameters(prior, alpha_prior_shape, alpha_prior_rate)
    gamma = trues(p)
    theta = 0.5
    theta_ref = Ref(theta)
    weights = _ssl_weights(prior, gamma)
    spike_weight = inv(prior.spike_scale)
    slab_weight = inv(prior.slab_scale)
    total_iter = warmup + n_samples * thin
    names = vcat(["beta[$j]" for j in 1:p], ["eta"], ["gamma[$j]" for j in 1:p], ["theta"])
    draws = Matrix{Float64}(undef, n_samples, length(names))
    lps = Vector{Float64}(undef, n_samples)
    u = vcat(zeros(Float64, p), [0.0])
    saved = 0
    accept_sum = 0.0
    accept_count = 0
    prog = _progress_meter(progress, total_iter, "SSLASSO explicit prox NUTS-Gibbs"; progress_desc = progress_desc, progress_offset = progress_offset)
    for iter in 1:total_iter
        theta_ref[] = theta
        target = CoxProxSSLassoTarget(
            data,
            prior,
            gamma,
            theta_ref;
            constraint = constraint,
            soft_center_scale = soft_center_scale,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = a_shape,
            alpha_prior_rate = a_rate,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
        )
        block = _sample_explicit_nuts_one(
            target;
            n_samples = 1,
            warmup = 0,
            rng = rng,
            target_accept = target_accept,
            max_depth = max_depth,
            progress = false,
            initial = u,
        )
        @views u .= block.draws[1, :]
        if isfinite(block.acceptance_rate)
            accept_sum += block.acceptance_rate
            accept_count += 1
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
            logit_p = log(theta_clamped) - log1p(-theta_clamped) - (h1 - h0)
            gamma[j] = rand(rng) < _logistic(logit_p)
            weights[j] = gamma[j] ? slab_weight : spike_weight
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
                soft_center_scale = soft_center_scale,
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
    acc = accept_count == 0 ? NaN : accept_sum / accept_count
    return MHSamples(draws, lps, names, acc, NaN, fill(NaN, length(names)))
end

function _sample_nuts_explicit(
    data::CoxPreparedData,
    prior::AbstractCoxPrior;
    n_samples::Integer = 5000,
    warmup::Integer = 2000,
    thin::Integer = 1,
    chains::Integer = 4,
    target_accept::Real = 0.8,
    max_depth::Integer = 10,
    rng::AbstractRNG = Random.default_rng(),
    initial = nothing,
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
    multichains::Bool = false,
    parallel_chains::Union{Nothing,Bool} = nothing,
    kwargs...,
)
    thin == 1 ||
        throw(ArgumentError("nuts_backend=:explicit currently expects thin=1; thin Turing NUTS support is unchanged."))
    chains > 0 || throw(ArgumentError("chains must be positive."))
    _validate_prior(prior)
    p = size(data.z, 2)
    resolved_constraint = _resolve_constraint(constraint, soft_center)
    scale = soft_center_scale === nothing ? _default_soft_center_scale(p) : Float64(soft_center_scale)
    _check_positive(prox_sum0_lambda, "prox_sum0_lambda")
    _check_positive(prox_l1_lambda, "prox_l1_lambda")
    _check_positive(prox_root_tol, "prox_root_tol")
    prox_root_maxiter > 0 || throw(ArgumentError("prox_root_maxiter must be positive."))
    a_shape, a_rate = _prox_alpha_hyperparameters(prior, alpha_prior_shape, alpha_prior_rate)
    generic_initial = _generic_explicit_initial(prior, data, initial)
    run_parallel = chains > 1 && something(parallel_chains, multichains || chains > 1)

    timed = @timed begin
        if prior isa SpikeSlabLassoPrior && resolved_constraint == :prox
            return _sample_ssl_prox_explicit_nuts_gibbs(
                data,
                prior;
                n_samples = n_samples,
                warmup = warmup,
                thin = thin,
                chains = chains,
                target_accept = target_accept,
                max_depth = max_depth,
                rng = rng,
                constraint = resolved_constraint,
                soft_center_scale = scale,
                prox_sum0_lambda = prox_sum0_lambda,
                prox_l1_lambda = prox_l1_lambda,
                alpha_prior_shape = a_shape,
                alpha_prior_rate = a_rate,
                prox_root_tol = prox_root_tol,
                prox_root_maxiter = prox_root_maxiter,
                progress = progress,
                parallel_chains = run_parallel,
            )
        end
        if chains == 1
            target = _make_explicit_nuts_target(
                data,
                prior;
                constraint = resolved_constraint,
                soft_center_scale = scale,
                prox_sum0_lambda = prox_sum0_lambda,
                prox_l1_lambda = prox_l1_lambda,
                alpha_prior_shape = a_shape,
                alpha_prior_rate = a_rate,
                prox_root_tol = prox_root_tol,
                prox_root_maxiter = prox_root_maxiter,
                car_weights = car_weights,
                use_car = use_car,
                car_precision = car_precision,
            )
            _sample_explicit_nuts_one(
                target;
                n_samples = n_samples,
                warmup = warmup,
                rng = rng,
                target_accept = target_accept,
                max_depth = max_depth,
                progress = progress,
                initial = prior isa BayesianLassoPrior && resolved_constraint == :prox ? initial : generic_initial,
            )
        else
            samples = Vector{MHSamples}(undef, chains)
            rngs = [_independent_rng(rng) for _ in 1:chains]
            if run_parallel
                _reserve_chain_progress_lines(chains, progress)
                Threads.@threads for c in 1:chains
                    @info "Running explicit NUTS chain $c/$chains"
                    target = _make_explicit_nuts_target(
                        data,
                        prior;
                        constraint = resolved_constraint,
                        soft_center_scale = scale,
                        prox_sum0_lambda = prox_sum0_lambda,
                        prox_l1_lambda = prox_l1_lambda,
                        alpha_prior_shape = a_shape,
                        alpha_prior_rate = a_rate,
                        prox_root_tol = prox_root_tol,
                        prox_root_maxiter = prox_root_maxiter,
                        car_weights = car_weights,
                        use_car = use_car,
                        car_precision = car_precision,
                    )
                    samples[c] = _sample_explicit_nuts_one(
                        target;
                        n_samples = n_samples,
                        warmup = warmup,
                        rng = rngs[c],
                        target_accept = target_accept,
                        max_depth = max_depth,
                        progress = progress,
                        progress_desc = _chain_progress_desc("AdvancedHMC NUTS sampling", c, chains),
                        progress_offset = _chain_progress_offset(c, chains),
                        initial = prior isa BayesianLassoPrior && resolved_constraint == :prox ? initial : generic_initial,
                    )
                end
            else
                for c in 1:chains
                    @info "Running explicit NUTS chain $c/$chains"
                    target = _make_explicit_nuts_target(
                        data,
                        prior;
                        constraint = resolved_constraint,
                        soft_center_scale = scale,
                        prox_sum0_lambda = prox_sum0_lambda,
                        prox_l1_lambda = prox_l1_lambda,
                        alpha_prior_shape = a_shape,
                        alpha_prior_rate = a_rate,
                        prox_root_tol = prox_root_tol,
                        prox_root_maxiter = prox_root_maxiter,
                        car_weights = car_weights,
                        use_car = use_car,
                        car_precision = car_precision,
                    )
                    samples[c] = _sample_explicit_nuts_one(
                        target;
                        n_samples = n_samples,
                        warmup = warmup,
                        rng = rngs[c],
                        target_accept = target_accept,
                        max_depth = max_depth,
                        progress = progress,
                        initial = prior isa BayesianLassoPrior && resolved_constraint == :prox ? initial : generic_initial,
                    )
                end
            end
            MultiMHSamples(samples)
        end
    end
    gc_percent = timed.time > 0 ? 100 * timed.gctime / timed.time : NaN
    diagnostics = SamplingDiagnostics(missing, missing, timed.time, Int64(timed.bytes), timed.gctime, gc_percent)
    return (samples = timed.value, diagnostics = diagnostics)
end
