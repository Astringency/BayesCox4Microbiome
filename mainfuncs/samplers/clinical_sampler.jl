"""Fit joint taxon/clinical Cox coefficients with explicit HMC.

Clinical columns must be coded before calling (e.g. binary indicators). Their
independent Student-t priors do not enter the taxon zero-sum or CAR penalties.
"""
function sample_clinical_cox(X, time, event, clinical_X;
    prior::AbstractCoxPrior, clinical_names = ["clinical[$j]" for j in axes(clinical_X, 2)],
    clinical_prior_df::Real = 3.0, clinical_prior_scale::Real = 2.5,
    n_samples::Integer = 5000, warmup::Integer = 2000, thin::Integer = 1,
    chains::Integer = 4, multichains::Bool = true, seed::Integer = 1,
    target_accept::Real = 0.8, hmc_n_leapfrog::Integer = 100, hmc_step_size = nothing,
    noncentered::Bool = true, constraint::Symbol = :prox, prox_sum0_lambda::Real = 1e-3,
    standardize::Bool = false, tie_method::Symbol = :efron,
    use_car::Bool = false, car_weights = nothing, car_distance = :bray,
    car_weight_quantile::Real = 0.99, car_precision::Real = 1.0,
    sample_car_strength::Bool = false, car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0, car_degree_scale::Symbol = :mean_active,
    car_kappa_init::Real = 0.5, prox_l1_lambda::Real = 1e-4,
    alpha_prior_shape = nothing, alpha_prior_scale = nothing, alpha_prior_kind = :invgamma,
    prox_root_tol::Real = 1e-10, prox_root_maxiter::Integer = 100,
    prox_block_step_size = nothing, prox_block_min_step_size::Real = 1e-8,
    prox_block_max_step_size::Real = 1.0, prox_block_adapt_every::Integer = 50,
    prox_block_divergence_threshold::Real = 1000.0, prox_block_metric = :unit_diag,
    init::Symbol = :zero, progress::Bool = false)
    n_samples > 0 && warmup >= 0 && thin > 0 && chains > 0 || throw(ArgumentError("invalid sampling counts"))
    prior isa Union{HorseshoePrior,RegularizedHorseshoePrior,SpikeSlabLassoPrior} ||
        throw(ArgumentError("joint clinical sampling supports HS/RHS and Prox-SSLASSO"))
    constraint in (:prox, :soft_center, :none) || throw(ArgumentError("unsupported clinical constraint"))
    all(startswith("clinical["), clinical_names) || throw(ArgumentError("clinical names must start with clinical["))
    prior isa SpikeSlabLassoPrior && use_car && throw(ArgumentError("Prox-SSLASSO has no CAR prior"))
    data = prepare_cox_data(X, time, event; standardize, tie_method)
    weights = !use_car ? nothing : car_weights !== nothing ? _prepare_car_weights(car_weights, size(X, 2)) :
        _car_weight_edges(car_weight_matrix(X; distance = car_distance, threshold_quantile = car_weight_quantile))
    rng = MersenneTwister(seed)
    rngs = chains == 1 ? [rng] : [_independent_rng(rng) for _ in 1:chains]
    phase_clock = _SamplingPhaseClock(chains)
    run_parallel = multichains && chains > 1 && Threads.nthreads() > 1
    function run_chain(c)
        if prior isa SpikeSlabLassoPrior
            return _sample_ssl_prox_explicit_hmc_gibbs_one(data, prior;
                n_samples, warmup, thin, rng = rngs[c], target_accept, hmc_n_leapfrog, hmc_step_size,
                prox_block_step_size, prox_block_min_step_size, prox_block_max_step_size,
                prox_block_adapt_every, prox_block_divergence_threshold, prox_block_metric,
                constraint, prox_sum0_lambda, prox_l1_lambda, alpha_prior_shape, alpha_prior_scale,
                alpha_prior_kind, prox_root_tol, prox_root_maxiter, init, progress,
                clinical_X, clinical_names, clinical_prior_df, clinical_prior_scale,
                phase_clock, chain_id = c)
        end
        base = _make_explicit_nuts_target(data, prior; constraint, prox_sum0_lambda, noncentered,
            car_weights = weights, use_car, car_precision, sample_car_strength,
            car_strength_prior_a, car_strength_prior_b, car_degree_scale, car_kappa_init)
        target = CoxClinicalTarget(base, clinical_X; names = clinical_names,
            prior_df = clinical_prior_df, prior_scale = clinical_prior_scale)
        return _sample_explicit_hmc_one(target; n_samples, warmup, thin,
            rng = rngs[c], target_accept, hmc_n_leapfrog, hmc_step_size, progress,
            initial = initial_state(target, rngs[c]), phase_clock, chain_id = c)
    end
    timed = @timed begin
        _start_sampling_clock!(phase_clock)
        output = Vector{MHSamples}(undef, chains)
        if run_parallel
            Threads.@threads for c in 1:chains
                output[c] = run_chain(c)
            end
        else
            for c in 1:chains
                output[c] = run_chain(c)
            end
        end
        chains == 1 ? first(output) : MultiMHSamples(output)
    end
    diag = _phase_diagnostics(timed, phase_clock, warmup, run_parallel)
    return CoxSamplerResult(timed.value, data, prior, :hmc, diag)
end
