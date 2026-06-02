struct ProxSSLassoBlockTransition
    accepted::Bool
    accept_prob::Float64
    logdensity::Float64
    energy_error::Float64
    divergent::Bool
    depth::Int
    n_leapfrog::Int
    step_size::Float64
end

mutable struct ProxSSLassoBlockHMCState
    step_size::Float64
    n_leapfrog::Int
    inv_metric::Vector{Float64}
    target_accept::Float64
    adapt_every::Int
    min_step_size::Float64
    max_step_size::Float64
    divergence_threshold::Float64
    accept_window::Int
    iter::Int
    frozen::Bool
    window_accept_sum::Float64
    window_accept_count::Int
    warmup_accept_sum::Float64
    warmup_accept_count::Int
    sample_accept_sum::Float64
    sample_accept_count::Int
    divergence_count::Int
    n_adapts::Int
    auto_step_size::Bool
    advanced_hamiltonian::Any
    advanced_kernel::Any
    advanced_adaptor::Any
    advanced_target::Any
end

mutable struct ProxSSLassoBlockNUTSState
    step_size::Float64
    max_depth::Int
    inv_metric::Vector{Float64}
    target_accept::Float64
    adapt_every::Int
    min_step_size::Float64
    max_step_size::Float64
    divergence_threshold::Float64
    accept_window::Int
    iter::Int
    frozen::Bool
    window_accept_sum::Float64
    window_accept_count::Int
    warmup_accept_sum::Float64
    warmup_accept_count::Int
    sample_accept_sum::Float64
    sample_accept_count::Int
    divergence_count::Int
    n_adapts::Int
    auto_step_size::Bool
    advanced_hamiltonian::Any
    advanced_kernel::Any
    advanced_adaptor::Any
    advanced_target::Any
end

function ProxSSLassoBlockHMCState(
    d::Integer;
    step_size::Real = 1e-3,
    n_leapfrog::Integer = 10,
    inv_metric = ones(Float64, Int(d)),
    target_accept::Real = 0.8,
    adapt_every::Integer = 50,
    min_step_size::Real = 1e-8,
    max_step_size::Real = 1.0,
    divergence_threshold::Real = 1000.0,
    frozen::Bool = false,
    n_adapts::Integer = 0,
    auto_step_size::Bool = false,
)
    d > 0 || throw(ArgumentError("block dimension must be positive."))
    n_leapfrog > 0 || throw(ArgumentError("n_leapfrog must be positive."))
    0 < target_accept < 1 || throw(ArgumentError("target_accept must be in (0, 1)."))
    adapt_every > 0 || throw(ArgumentError("prox_block_adapt_every must be positive."))
    n_adapts >= 0 || throw(ArgumentError("n_adapts must be nonnegative."))
    _check_positive(divergence_threshold, "prox_block_divergence_threshold")
    return ProxSSLassoBlockHMCState(
        _prox_ssl_checked_step_size(step_size, min_step_size, max_step_size),
        Int(n_leapfrog),
        _prox_ssl_block_inv_metric(Int(d), inv_metric),
        Float64(target_accept),
        Int(adapt_every),
        Float64(min_step_size),
        Float64(max_step_size),
        Float64(divergence_threshold),
        Int(adapt_every),
        0,
        frozen,
        0.0,
        0,
        0.0,
        0,
        0.0,
        0,
        0,
        Int(n_adapts),
        auto_step_size,
        nothing,
        nothing,
        nothing,
        nothing,
    )
end

function ProxSSLassoBlockNUTSState(
    d::Integer;
    step_size::Real = 1e-3,
    max_depth::Integer = 10,
    inv_metric = ones(Float64, Int(d)),
    target_accept::Real = 0.8,
    adapt_every::Integer = 50,
    min_step_size::Real = 1e-8,
    max_step_size::Real = 1.0,
    divergence_threshold::Real = 1000.0,
    frozen::Bool = false,
    n_adapts::Integer = 0,
    auto_step_size::Bool = false,
)
    d > 0 || throw(ArgumentError("block dimension must be positive."))
    max_depth > 0 || throw(ArgumentError("max_depth must be positive."))
    0 < target_accept < 1 || throw(ArgumentError("target_accept must be in (0, 1)."))
    adapt_every > 0 || throw(ArgumentError("prox_block_adapt_every must be positive."))
    n_adapts >= 0 || throw(ArgumentError("n_adapts must be nonnegative."))
    _check_positive(divergence_threshold, "prox_block_divergence_threshold")
    return ProxSSLassoBlockNUTSState(
        _prox_ssl_checked_step_size(step_size, min_step_size, max_step_size),
        Int(max_depth),
        _prox_ssl_block_inv_metric(Int(d), inv_metric),
        Float64(target_accept),
        Int(adapt_every),
        Float64(min_step_size),
        Float64(max_step_size),
        Float64(divergence_threshold),
        Int(adapt_every),
        0,
        frozen,
        0.0,
        0,
        0.0,
        0,
        0.0,
        0,
        0,
        Int(n_adapts),
        auto_step_size,
        nothing,
        nothing,
        nothing,
        nothing,
    )
end

function _prox_ssl_checked_step_size(step_size, min_step_size, max_step_size)
    _check_positive(step_size, "prox block step_size")
    _check_positive(min_step_size, "prox_block_min_step_size")
    _check_positive(max_step_size, "prox_block_max_step_size")
    min_step_size <= max_step_size ||
        throw(ArgumentError("prox_block_min_step_size must be <= prox_block_max_step_size."))
    return clamp(Float64(step_size), Float64(min_step_size), Float64(max_step_size))
end

function _prox_ssl_block_inv_metric(d::Int, metric::Symbol)
    metric == :unit_diag ||
        throw(ArgumentError("prox_block_metric currently supports :unit_diag or a positive diagonal vector."))
    return ones(Float64, d)
end

function _prox_ssl_block_inv_metric(d::Int, metric::AbstractVector)
    length(metric) == d || throw(DimensionMismatch("prox block metric length must match block dimension."))
    inv_metric = collect(Float64, metric)
    all(x -> isfinite(x) && x > 0, inv_metric) ||
        throw(ArgumentError("prox block metric entries must be finite and positive."))
    return inv_metric
end

function _prox_ssl_block_inv_metric(d::Int, metric)
    metric === nothing && return ones(Float64, d)
    return _prox_ssl_block_inv_metric(d, metric)
end

function _prox_ssl_block_adapt_step_size!(state, window_accept_rate::Real)
    state.frozen && return state.step_size
    rate = Float64(window_accept_rate)
    isfinite(rate) || return state.step_size
    delta = clamp(rate - state.target_accept, -0.5, 0.5)
    state.step_size = clamp(
        state.step_size * exp(delta),
        state.min_step_size,
        state.max_step_size,
    )
    return state.step_size
end

function _prox_ssl_block_record_transition!(state, accept_prob::Real, divergent::Bool)
    ap = isfinite(accept_prob) ? clamp(Float64(accept_prob), 0.0, 1.0) : 0.0
    state.iter += 1
    divergent && (state.divergence_count += 1)
    if state.frozen
        state.sample_accept_sum += ap
        state.sample_accept_count += 1
        return nothing
    end
    state.warmup_accept_sum += ap
    state.warmup_accept_count += 1
    state.window_accept_sum += ap
    state.window_accept_count += 1
    if state.window_accept_count >= state.accept_window
        if state.advanced_adaptor === nothing
            _prox_ssl_block_adapt_step_size!(
                state,
                state.window_accept_sum / state.window_accept_count,
            )
        end
        state.window_accept_sum = 0.0
        state.window_accept_count = 0
    end
    return nothing
end

function _prox_ssl_block_freeze!(state)
    if !state.frozen && state.advanced_adaptor !== nothing
        AdvancedHMC.Adaptation.finalize!(state.advanced_adaptor)
        state.advanced_hamiltonian = AdvancedHMC.update(state.advanced_hamiltonian, state.advanced_adaptor)
        state.advanced_kernel = AdvancedHMC.update(state.advanced_kernel, state.advanced_adaptor)
        _prox_ssl_block_sync_from_advancedhmc!(state)
    elseif !state.frozen && state.window_accept_count > 0
        _prox_ssl_block_adapt_step_size!(
            state,
            state.window_accept_sum / state.window_accept_count,
        )
    end
    state.window_accept_sum = 0.0
    state.window_accept_count = 0
    state.frozen = true
    return state
end

_prox_ssl_block_sampling_acceptance(state) =
    state.sample_accept_count == 0 ? NaN : state.sample_accept_sum / state.sample_accept_count

_prox_ssl_block_warmup_acceptance(state) =
    state.warmup_accept_count == 0 ? NaN : state.warmup_accept_sum / state.warmup_accept_count

_prox_ssl_block_stat_get(stat::NamedTuple, key::Symbol, default) =
    haskey(stat, key) ? getproperty(stat, key) : default

function _prox_ssl_block_make_hamiltonian(target::AbstractCoxTarget, inv_metric::AbstractVector)
    logp = u -> logdensity(target, u)
    gradlogp = u -> LogDensityProblems.logdensity_and_gradient(target, u)
    metric = AdvancedHMC.DiagEuclideanMetric(copy(inv_metric))
    return AdvancedHMC.Hamiltonian(metric, logp, gradlogp)
end

function _prox_ssl_block_make_kernel(state::ProxSSLassoBlockHMCState)
    integrator = AdvancedHMC.Leapfrog(state.step_size)
    return AdvancedHMC.HMCKernel(
        AdvancedHMC.Trajectory{AdvancedHMC.EndPointTS}(
            integrator,
            AdvancedHMC.FixedNSteps(state.n_leapfrog),
        ),
    )
end

function _prox_ssl_block_make_kernel(state::ProxSSLassoBlockNUTSState)
    integrator = AdvancedHMC.Leapfrog(state.step_size)
    return AdvancedHMC.HMCKernel(
        AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(
            integrator,
            AdvancedHMC.GeneralisedNoUTurn(state.max_depth, state.divergence_threshold),
        ),
    )
end

function _prox_ssl_block_set_adaptor_step_size!(state)
    state.advanced_adaptor === nothing && return nothing
    step_adaptor = getfield(state.advanced_adaptor, :ssa)
    hasfield(typeof(step_adaptor), :state) || return nothing
    setfield!(getfield(step_adaptor, :state), 2, state.step_size)
    return nothing
end

function _prox_ssl_block_sync_from_advancedhmc!(state)
    state.advanced_kernel === nothing && return nothing
    trajectory = getfield(state.advanced_kernel, 2)
    integrator = getfield(trajectory, 1)
    raw_step = Float64(AdvancedHMC.nom_step_size(integrator))
    state.step_size = _prox_ssl_checked_step_size(
        raw_step,
        state.min_step_size,
        state.max_step_size,
    )
    if state.step_size != raw_step
        _prox_ssl_block_set_adaptor_step_size!(state)
        state.advanced_kernel = _prox_ssl_block_make_kernel(state)
    end
    if state.advanced_hamiltonian !== nothing
        metric = getfield(state.advanced_hamiltonian, 1)
        metric_inv = getfield(metric, 1)
        if length(metric_inv) == length(state.inv_metric)
            state.inv_metric .= metric_inv
        end
    end
    return nothing
end

function _prox_ssl_block_initialize_advancedhmc!(
    state,
    target::AbstractCoxTarget,
    u::AbstractVector{Float64},
    rng::AbstractRNG,
)
    if state.advanced_hamiltonian !== nothing && state.advanced_target === target
        return nothing
    end
    length(state.inv_metric) == length(u) ||
        throw(DimensionMismatch("prox block metric length must match current block."))
    state.advanced_target = target
    state.advanced_hamiltonian = _prox_ssl_block_make_hamiltonian(target, state.inv_metric)
    if state.auto_step_size
        found_step = AdvancedHMC.find_good_stepsize(rng, state.advanced_hamiltonian, u)
        state.step_size = _prox_ssl_checked_step_size(
            found_step,
            state.min_step_size,
            state.max_step_size,
        )
        state.auto_step_size = false
    end
    state.advanced_kernel = _prox_ssl_block_make_kernel(state)
    trajectory = getfield(state.advanced_kernel, 2)
    integrator = getfield(trajectory, 1)
    metric = getfield(state.advanced_hamiltonian, 1)
    state.advanced_adaptor = AdvancedHMC.StanHMCAdaptor(
        AdvancedHMC.MassMatrixAdaptor(metric),
        AdvancedHMC.StepSizeAdaptor(Float64(state.target_accept), integrator),
    )
    return nothing
end

function _prox_ssl_block_advancedhmc_step!(
    u::Vector{Float64},
    target::AbstractCoxTarget,
    state;
    rng::AbstractRNG,
)
    _prox_ssl_block_initialize_advancedhmc!(state, target, u, rng)
    z0 = AdvancedHMC.phasepoint(
        state.advanced_hamiltonian,
        copy(u),
        zeros(Float64, length(u)),
    )
    isfinite(z0) ||
        throw(ArgumentError("current AdvancedHMC block state has non-finite logdensity or gradient."))

    transition = AdvancedHMC.transition(
        rng,
        state.advanced_hamiltonian,
        state.advanced_kernel,
        z0,
    )
    stat = AdvancedHMC.stat(transition)
    position = getfield(transition.z, 1)
    u .= position

    accept_prob = Float64(_prox_ssl_block_stat_get(stat, :acceptance_rate, 0.0))
    divergent = Bool(_prox_ssl_block_stat_get(stat, :numerical_error, false))
    if !state.frozen && state.n_adapts > 0
        state.advanced_hamiltonian, state.advanced_kernel, _ = AdvancedHMC.Adaptation.adapt!(
            state.advanced_hamiltonian,
            state.advanced_kernel,
            state.advanced_adaptor,
            state.iter + 1,
            state.n_adapts,
            position,
            accept_prob,
        )
    end
    _prox_ssl_block_sync_from_advancedhmc!(state)
    _prox_ssl_block_record_transition!(state, accept_prob, divergent)

    return ProxSSLassoBlockTransition(
        Bool(_prox_ssl_block_stat_get(stat, :is_accept, true)),
        accept_prob,
        Float64(_prox_ssl_block_stat_get(stat, :log_density, logdensity(target, u))),
        Float64(_prox_ssl_block_stat_get(
            stat,
            :max_hamiltonian_energy_error,
            _prox_ssl_block_stat_get(stat, :hamiltonian_energy_error, NaN),
        )),
        divergent,
        Int(_prox_ssl_block_stat_get(stat, :tree_depth, 0)),
        Int(_prox_ssl_block_stat_get(stat, :n_steps, 0)),
        state.step_size,
    )
end

function prox_ssl_block_hmc_step!(
    u::Vector{Float64},
    target::AbstractCoxTarget,
    state::ProxSSLassoBlockHMCState;
    rng::AbstractRNG,
)
    return _prox_ssl_block_advancedhmc_step!(u, target, state; rng = rng)
end

function prox_ssl_block_nuts_step!(
    u::Vector{Float64},
    target::AbstractCoxTarget,
    state::ProxSSLassoBlockNUTSState;
    rng::AbstractRNG,
)
    return _prox_ssl_block_advancedhmc_step!(u, target, state; rng = rng)
end
