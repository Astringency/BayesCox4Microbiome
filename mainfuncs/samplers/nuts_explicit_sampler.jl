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

    logπ = u -> logdensity(target, u)
    gradπ = u -> LogDensityProblems.logdensity_and_gradient(target, u)
    metric = AdvancedHMC.DiagEuclideanMetric(d)
    hamiltonian = AdvancedHMC.Hamiltonian(metric, logπ, gradπ)
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
)
    constraint == :prox ||
        throw(ArgumentError("nuts_backend=:explicit currently supports BayesianLassoPrior only with constraint=:prox."))
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
    kwargs...,
)
    throw(ArgumentError(
        "nuts_backend=:explicit currently supports prox-LASSO with BayesianLassoPrior. " *
        "Use nuts_backend=:turing for $(typeof(prior)).",
    ))
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
    constraint::Symbol = :soft_center,
    soft_center = nothing,
    soft_center_scale::Union{Nothing,Real} = nothing,
    prox_sum0_lambda::Real = 1e-3,
    prox_l1_lambda::Real = 1e-3,
    alpha_prior_shape = nothing,
    alpha_prior_rate = nothing,
    prox_root_tol::Real = 1e-10,
    prox_root_maxiter::Integer = 100,
    progress::Bool = true,
    multichains::Bool = false,
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

    timed = @timed begin
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
            )
            _sample_explicit_nuts_one(
                target;
                n_samples = n_samples,
                warmup = warmup,
                rng = rng,
                target_accept = target_accept,
                max_depth = max_depth,
                progress = progress,
            )
        else
            samples = Vector{MHSamples}(undef, chains)
            rngs = [_independent_rng(rng) for _ in 1:chains]
            if multichains
                Threads.@threads for c in 1:chains
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
                    )
                    samples[c] = _sample_explicit_nuts_one(
                        target;
                        n_samples = n_samples,
                        warmup = warmup,
                        rng = rngs[c],
                        target_accept = target_accept,
                        max_depth = max_depth,
                        progress = false,
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
                    )
                    samples[c] = _sample_explicit_nuts_one(
                        target;
                        n_samples = n_samples,
                        warmup = warmup,
                        rng = rngs[c],
                        target_accept = target_accept,
                        max_depth = max_depth,
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
