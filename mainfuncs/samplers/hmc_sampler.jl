function _sample_explicit_hmc_one(
    target::AbstractCoxTarget;
    n_samples::Integer,
    warmup::Integer,
    thin::Integer,
    rng::AbstractRNG,
    target_accept::Real,
    hmc_n_leapfrog::Integer,
    hmc_step_size = nothing,
    progress::Bool,
    progress_desc::AbstractString = "HMC sampling",
    progress_offset::Integer = 0,
    initial = nothing,
)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive."))
    warmup >= 0 || throw(ArgumentError("warmup must be nonnegative."))
    thin > 0 || throw(ArgumentError("thin must be positive."))
    hmc_n_leapfrog > 0 || throw(ArgumentError("hmc_n_leapfrog must be positive."))
    0 < target_accept < 1 || throw(ArgumentError("target_accept must be in (0, 1)."))
    hmc_step_size === nothing || _check_positive(hmc_step_size, "hmc_step_size")

    init = initial === nothing ? initial_state(target) : collect(Float64, initial)
    d = dimension(target)
    length(init) == d || throw(DimensionMismatch("initial length must match target dimension."))
    isfinite(logdensity(target, init)) ||
        throw(ArgumentError("initial state has non-finite log posterior."))

    logp = u -> logdensity(target, u)
    gradlogp = u -> LogDensityProblems.logdensity_and_gradient(target, u)
    metric = AdvancedHMC.DiagEuclideanMetric(d)
    hamiltonian = AdvancedHMC.Hamiltonian(metric, logp, gradlogp)
    initial_step = hmc_step_size === nothing ?
        AdvancedHMC.find_good_stepsize(rng, hamiltonian, init) :
        Float64(hmc_step_size)
    integrator = AdvancedHMC.Leapfrog(initial_step)
    kernel = AdvancedHMC.HMCKernel(
        AdvancedHMC.Trajectory{AdvancedHMC.EndPointTS}(
            integrator,
            AdvancedHMC.FixedNSteps(Int(hmc_n_leapfrog)),
        ),
    )
    adaptor = AdvancedHMC.StanHMCAdaptor(
        AdvancedHMC.MassMatrixAdaptor(metric),
        AdvancedHMC.StepSizeAdaptor(Float64(target_accept), integrator),
    )

    saved_iter = Int(n_samples) * Int(thin)
    total_iter = warmup + saved_iter
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

    length(samples) == saved_iter ||
        throw(ArgumentError("AdvancedHMC returned an unexpected number of post-warmup states."))
    output_names = sample_state_names(target)
    draws = Matrix{Float64}(undef, n_samples, length(output_names))
    lps = Vector{Float64}(undef, n_samples)
    @inbounds for i in 1:n_samples
        state = samples[i * Int(thin)]
        draws[i, :] .= sample_state(target, state)
        lps[i] = sample_logdensity(target, view(draws, i, :), state)
    end
    return MHSamples(
        draws,
        lps,
        output_names,
        _advancedhmc_acceptance_rate(stats),
        NaN,
        fill(NaN, length(output_names)),
    )
end

function _sample_ssl_prox_explicit_hmc_gibbs(
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior;
    n_samples::Integer,
    warmup::Integer,
    thin::Integer,
    chains::Integer,
    target_accept::Real,
    hmc_n_leapfrog::Integer,
    hmc_step_size,
    prox_block_step_size,
    prox_block_min_step_size::Real,
    prox_block_max_step_size::Real,
    prox_block_adapt_every::Integer,
    prox_block_divergence_threshold::Real,
    prox_block_metric,
    rng::AbstractRNG,
    constraint::Symbol,
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
            _sample_ssl_prox_explicit_hmc_gibbs_one(
                data,
                prior;
                n_samples = n_samples,
                warmup = warmup,
                thin = thin,
                rng = rng,
                target_accept = target_accept,
                hmc_n_leapfrog = hmc_n_leapfrog,
                hmc_step_size = hmc_step_size,
                prox_block_step_size = prox_block_step_size,
                prox_block_min_step_size = prox_block_min_step_size,
                prox_block_max_step_size = prox_block_max_step_size,
                prox_block_adapt_every = prox_block_adapt_every,
                prox_block_divergence_threshold = prox_block_divergence_threshold,
                prox_block_metric = prox_block_metric,
                constraint = constraint,
                prox_sum0_lambda = prox_sum0_lambda,
                prox_l1_lambda = prox_l1_lambda,
                alpha_prior_shape = alpha_prior_shape,
                alpha_prior_rate = alpha_prior_rate,
                prox_root_tol = prox_root_tol,
                prox_root_maxiter = prox_root_maxiter,
                progress = progress,
                progress_desc = "SSLASSO explicit prox HMC-Gibbs",
                progress_offset = 0,
            )
        else
            samples = Vector{MHSamples}(undef, chains)
            rngs = [_independent_rng(rng) for _ in 1:chains]
            if parallel_chains
                _reserve_chain_progress_lines(chains, progress)
                Threads.@threads for c in 1:chains
                    samples[c] = _sample_ssl_prox_explicit_hmc_gibbs_one(
                        data,
                        prior;
                        n_samples = n_samples,
                        warmup = warmup,
                        thin = thin,
                        rng = rngs[c],
                        target_accept = target_accept,
                        hmc_n_leapfrog = hmc_n_leapfrog,
                        hmc_step_size = hmc_step_size,
                        prox_block_step_size = prox_block_step_size,
                        prox_block_min_step_size = prox_block_min_step_size,
                        prox_block_max_step_size = prox_block_max_step_size,
                        prox_block_adapt_every = prox_block_adapt_every,
                        prox_block_divergence_threshold = prox_block_divergence_threshold,
                        prox_block_metric = prox_block_metric,
                        constraint = constraint,
                        prox_sum0_lambda = prox_sum0_lambda,
                        prox_l1_lambda = prox_l1_lambda,
                        alpha_prior_shape = alpha_prior_shape,
                        alpha_prior_rate = alpha_prior_rate,
                        prox_root_tol = prox_root_tol,
                        prox_root_maxiter = prox_root_maxiter,
                        progress = progress,
                        progress_desc = _chain_progress_desc("SSLASSO explicit prox HMC-Gibbs", c, chains),
                        progress_offset = _chain_progress_offset(c, chains),
                    )
                end
            else
                for c in 1:chains
                    @info "Running explicit SSLASSO prox HMC-Gibbs chain $c/$chains"
                    samples[c] = _sample_ssl_prox_explicit_hmc_gibbs_one(
                        data,
                        prior;
                        n_samples = n_samples,
                        warmup = warmup,
                        thin = thin,
                        rng = rngs[c],
                        target_accept = target_accept,
                        hmc_n_leapfrog = hmc_n_leapfrog,
                        hmc_step_size = hmc_step_size,
                        prox_block_step_size = prox_block_step_size,
                        prox_block_min_step_size = prox_block_min_step_size,
                        prox_block_max_step_size = prox_block_max_step_size,
                        prox_block_adapt_every = prox_block_adapt_every,
                        prox_block_divergence_threshold = prox_block_divergence_threshold,
                        prox_block_metric = prox_block_metric,
                        constraint = constraint,
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

function _sample_ssl_prox_explicit_hmc_gibbs_one(
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior;
    n_samples::Integer,
    warmup::Integer,
    thin::Integer,
    rng::AbstractRNG,
    target_accept::Real,
    hmc_n_leapfrog::Integer,
    hmc_step_size,
    prox_block_step_size,
    prox_block_min_step_size::Real,
    prox_block_max_step_size::Real,
    prox_block_adapt_every::Integer,
    prox_block_divergence_threshold::Real,
    prox_block_metric,
    constraint::Symbol,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape,
    alpha_prior_rate,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
    progress::Bool,
    progress_desc::AbstractString = "SSLASSO explicit prox HMC-Gibbs",
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
    initial_step = prox_block_step_size === nothing ?
        (hmc_step_size === nothing ? 1e-3 : hmc_step_size) :
        prox_block_step_size
    hmc_state = ProxSSLassoBlockHMCState(
        p + 1;
        step_size = initial_step,
        n_leapfrog = hmc_n_leapfrog,
        inv_metric = prox_block_metric,
        target_accept = target_accept,
        adapt_every = prox_block_adapt_every,
        min_step_size = prox_block_min_step_size,
        max_step_size = prox_block_max_step_size,
        divergence_threshold = prox_block_divergence_threshold,
        frozen = warmup == 0,
    )
    prog = _progress_meter(progress, total_iter, "SSLASSO explicit prox HMC-Gibbs"; progress_desc = progress_desc, progress_offset = progress_offset)
    for iter in 1:total_iter
        theta_ref[] = theta
        target = CoxProxSSLassoTarget(
            data,
            prior,
            gamma,
            theta_ref;
            constraint = constraint,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = a_shape,
            alpha_prior_rate = a_rate,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
        )
        transition = prox_ssl_block_hmc_step!(u, target, hmc_state; rng = rng)
        iter == warmup && _prox_ssl_block_freeze!(hmc_state)

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
                prox_sum0_lambda = prox_sum0_lambda,
                prox_l1_lambda = prox_l1_lambda,
                alpha_prior_shape = a_shape,
                alpha_prior_rate = a_rate,
                prox_root_tol = prox_root_tol,
                prox_root_maxiter = prox_root_maxiter,
            )
        end
        phase = iter <= warmup ? "warmup $(iter)/$(warmup)" : "sample $(saved)/$(n_samples)"
        _progress_next!(
            prog;
            showvalues = [
                (:phase, phase),
                (:saved, saved),
                (:step_size, transition.step_size),
                (:accept, transition.accept_prob),
            ],
        )
    end
    _progress_finish!(prog)
    final_step = fill(hmc_state.step_size, length(names))
    return MHSamples(
        draws,
        lps,
        names,
        _prox_ssl_block_sampling_acceptance(hmc_state),
        _prox_ssl_block_warmup_acceptance(hmc_state),
        final_step,
    )
end

function _sample_hmc_explicit(
    data::CoxPreparedData,
    prior::AbstractCoxPrior;
    n_samples::Integer = 5000,
    warmup::Integer = 2000,
    thin::Integer = 1,
    chains::Integer = 1,
    target_accept::Real = 0.8,
    hmc_n_leapfrog::Integer = 10,
    hmc_step_size = nothing,
    prox_block_step_size = nothing,
    prox_block_min_step_size::Real = 1e-8,
    prox_block_max_step_size::Real = 1.0,
    prox_block_adapt_every::Integer = 50,
    prox_block_divergence_threshold::Real = 1000.0,
    prox_block_metric = :unit_diag,
    rng::AbstractRNG = Random.default_rng(),
    initial = nothing,
    constraint::Symbol = :soft_center,
    prox_sum0_lambda::Real = 1e-3,
    prox_l1_lambda::Real = 1e-3,
    alpha_prior_shape = nothing,
    alpha_prior_rate = nothing,
    prox_root_tol::Real = 1e-10,
    prox_root_maxiter::Integer = 100,
    car_weights = nothing,
    use_car::Bool = car_weights !== nothing,
    car_precision::Real = 1.0,
    sample_car_strength::Bool = false,
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_degree_scale::Symbol = :mean_active,
    car_kappa_init::Real = 0.5,
    progress::Bool = true,
    multichains::Bool = false,
    parallel_chains::Union{Nothing,Bool} = nothing,
    noncentered::Bool = false,
    kwargs...,
)
    thin > 0 || throw(ArgumentError("thin must be positive."))
    chains > 0 || throw(ArgumentError("chains must be positive."))
    hmc_n_leapfrog > 0 || throw(ArgumentError("hmc_n_leapfrog must be positive."))
    hmc_step_size === nothing || _check_positive(hmc_step_size, "hmc_step_size")
    _validate_prior(prior)
    resolved_constraint = _resolve_constraint(constraint)
    _check_positive(prox_sum0_lambda, "prox_sum0_lambda")
    _check_positive(prox_l1_lambda, "prox_l1_lambda")
    _check_positive(prox_root_tol, "prox_root_tol")
    prox_root_maxiter > 0 || throw(ArgumentError("prox_root_maxiter must be positive."))
    a_shape, a_rate = _prox_alpha_hyperparameters(prior, alpha_prior_shape, alpha_prior_rate)
    if noncentered
        _check_noncentered_supported(prior)
    end
    effective_car_weights = use_car ? car_weights : nothing
    if sample_car_strength
        _validate_adaptive_car_options(
            effective_car_weights,
            car_strength_prior_a,
            car_strength_prior_b,
            car_kappa_init,
        )
    end
    generic_initial = prior isa Union{HorseshoePrior,RegularizedHorseshoePrior} ?
        _generic_explicit_initial(
            prior,
            data,
            initial;
            noncentered = noncentered,
            car_weights = effective_car_weights,
            sample_car_strength = sample_car_strength,
            car_strength_prior_a = car_strength_prior_a,
            car_strength_prior_b = car_strength_prior_b,
            car_kappa_init = car_kappa_init,
        ) :
        _generic_explicit_initial(
            prior,
            data,
            initial;
            car_weights = effective_car_weights,
            sample_car_strength = false,
        )
    run_parallel = chains > 1 && something(parallel_chains, multichains)
    use_proximal_prior = _is_proximal_l1_prior(prior)

    timed = @timed begin
        if prior isa SpikeSlabLassoPrior && use_proximal_prior
            return _sample_ssl_prox_explicit_hmc_gibbs(
                data,
                prior;
                n_samples = n_samples,
                warmup = warmup,
                thin = thin,
                chains = chains,
                target_accept = target_accept,
                hmc_n_leapfrog = hmc_n_leapfrog,
                hmc_step_size = hmc_step_size,
                prox_block_step_size = prox_block_step_size,
                prox_block_min_step_size = prox_block_min_step_size,
                prox_block_max_step_size = prox_block_max_step_size,
                prox_block_adapt_every = prox_block_adapt_every,
                prox_block_divergence_threshold = prox_block_divergence_threshold,
                prox_block_metric = prox_block_metric,
                rng = rng,
                constraint = resolved_constraint,
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
                prox_sum0_lambda = prox_sum0_lambda,
                prox_l1_lambda = prox_l1_lambda,
                alpha_prior_shape = a_shape,
                alpha_prior_rate = a_rate,
                prox_root_tol = prox_root_tol,
                prox_root_maxiter = prox_root_maxiter,
                car_weights = car_weights,
                use_car = use_car,
                car_precision = car_precision,
                sample_car_strength = sample_car_strength,
                car_strength_prior_a = car_strength_prior_a,
                car_strength_prior_b = car_strength_prior_b,
                car_degree_scale = car_degree_scale,
                car_kappa_init = car_kappa_init,
                noncentered = noncentered,
            )
            _sample_explicit_hmc_one(
                target;
                n_samples = n_samples,
                warmup = warmup,
                thin = thin,
                rng = rng,
                target_accept = target_accept,
                hmc_n_leapfrog = hmc_n_leapfrog,
                hmc_step_size = hmc_step_size,
                progress = progress,
                initial = prior isa BayesianLassoPrior && use_proximal_prior ? initial : generic_initial,
            )
        else
            samples = Vector{MHSamples}(undef, chains)
            rngs = [_independent_rng(rng) for _ in 1:chains]
            if run_parallel
                _reserve_chain_progress_lines(chains, progress)
                Threads.@threads for c in 1:chains
                    @info "Running explicit HMC chain $c/$chains"
                    target = _make_explicit_nuts_target(
                        data,
                        prior;
                        constraint = resolved_constraint,
                        prox_sum0_lambda = prox_sum0_lambda,
                        prox_l1_lambda = prox_l1_lambda,
                        alpha_prior_shape = a_shape,
                        alpha_prior_rate = a_rate,
                        prox_root_tol = prox_root_tol,
                        prox_root_maxiter = prox_root_maxiter,
                        car_weights = car_weights,
                        use_car = use_car,
                        car_precision = car_precision,
                        sample_car_strength = sample_car_strength,
                        car_strength_prior_a = car_strength_prior_a,
                        car_strength_prior_b = car_strength_prior_b,
                        car_degree_scale = car_degree_scale,
                        car_kappa_init = car_kappa_init,
                        noncentered = noncentered,
                    )
                    samples[c] = _sample_explicit_hmc_one(
                        target;
                        n_samples = n_samples,
                        warmup = warmup,
                        thin = thin,
                        rng = rngs[c],
                        target_accept = target_accept,
                        hmc_n_leapfrog = hmc_n_leapfrog,
                        hmc_step_size = hmc_step_size,
                        progress = progress,
                        progress_desc = _chain_progress_desc("AdvancedHMC HMC sampling", c, chains),
                        progress_offset = _chain_progress_offset(c, chains),
                        initial = prior isa BayesianLassoPrior && use_proximal_prior ? initial : generic_initial,
                    )
                end
            else
                for c in 1:chains
                    @info "Running explicit HMC chain $c/$chains"
                    target = _make_explicit_nuts_target(
                        data,
                        prior;
                        constraint = resolved_constraint,
                        prox_sum0_lambda = prox_sum0_lambda,
                        prox_l1_lambda = prox_l1_lambda,
                        alpha_prior_shape = a_shape,
                        alpha_prior_rate = a_rate,
                        prox_root_tol = prox_root_tol,
                        prox_root_maxiter = prox_root_maxiter,
                        car_weights = car_weights,
                        use_car = use_car,
                        car_precision = car_precision,
                        sample_car_strength = sample_car_strength,
                        car_strength_prior_a = car_strength_prior_a,
                        car_strength_prior_b = car_strength_prior_b,
                        car_degree_scale = car_degree_scale,
                        car_kappa_init = car_kappa_init,
                        noncentered = noncentered,
                    )
                    samples[c] = _sample_explicit_hmc_one(
                        target;
                        n_samples = n_samples,
                        warmup = warmup,
                        thin = thin,
                        rng = rngs[c],
                        target_accept = target_accept,
                        hmc_n_leapfrog = hmc_n_leapfrog,
                        hmc_step_size = hmc_step_size,
                        progress = progress,
                        initial = prior isa BayesianLassoPrior && use_proximal_prior ? initial : generic_initial,
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
