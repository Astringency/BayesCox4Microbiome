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
)
    d > 0 || throw(ArgumentError("block dimension must be positive."))
    n_leapfrog > 0 || throw(ArgumentError("n_leapfrog must be positive."))
    0 < target_accept < 1 || throw(ArgumentError("target_accept must be in (0, 1)."))
    adapt_every > 0 || throw(ArgumentError("prox_block_adapt_every must be positive."))
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
)
    d > 0 || throw(ArgumentError("block dimension must be positive."))
    max_depth > 0 || throw(ArgumentError("max_depth must be positive."))
    0 < target_accept < 1 || throw(ArgumentError("target_accept must be in (0, 1)."))
    adapt_every > 0 || throw(ArgumentError("prox_block_adapt_every must be positive."))
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

@inline function _prox_ssl_block_kinetic(momentum::AbstractVector, inv_metric::AbstractVector)
    s = 0.0
    @inbounds for i in eachindex(momentum, inv_metric)
        s += abs2(momentum[i]) * inv_metric[i]
    end
    return 0.5 * s
end

function _prox_ssl_block_sample_momentum!(momentum::AbstractVector, inv_metric::AbstractVector, rng::AbstractRNG)
    @inbounds for i in eachindex(momentum, inv_metric)
        momentum[i] = randn(rng) / sqrt(inv_metric[i])
    end
    return momentum
end

function _prox_ssl_block_logdensity_and_gradient!(
    grad::AbstractVector{Float64},
    target::AbstractCoxTarget,
    u::AbstractVector,
)
    lp, _ = logdensity_and_gradient!(grad, target, u)
    return Float64(lp), isfinite(lp) && all(isfinite, grad)
end

function _prox_ssl_block_leapfrog!(
    q::AbstractVector{Float64},
    momentum::AbstractVector{Float64},
    grad::AbstractVector{Float64},
    target::AbstractCoxTarget,
    inv_metric::AbstractVector{Float64},
    step_size::Float64,
)
    @inbounds for i in eachindex(momentum, grad)
        momentum[i] += 0.5 * step_size * grad[i]
        q[i] += step_size * inv_metric[i] * momentum[i]
    end
    lp, ok = _prox_ssl_block_logdensity_and_gradient!(grad, target, q)
    ok || return lp, false
    @inbounds for i in eachindex(momentum, grad)
        momentum[i] += 0.5 * step_size * grad[i]
    end
    return lp, true
end

@inline function _prox_ssl_block_accept_probability(energy_error::Float64)
    isfinite(energy_error) || return 0.0
    energy_error <= 0 && return 1.0
    return exp(-energy_error)
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
        _prox_ssl_block_adapt_step_size!(
            state,
            state.window_accept_sum / state.window_accept_count,
        )
        state.window_accept_sum = 0.0
        state.window_accept_count = 0
    end
    return nothing
end

function _prox_ssl_block_freeze!(state)
    if !state.frozen && state.window_accept_count > 0
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

function prox_ssl_block_hmc_step!(
    u::Vector{Float64},
    target::AbstractCoxTarget,
    state::ProxSSLassoBlockHMCState;
    rng::AbstractRNG,
)
    d = length(u)
    length(state.inv_metric) == d ||
        throw(DimensionMismatch("HMC block state metric length must match current block."))
    grad = Vector{Float64}(undef, d)
    current_lp, ok = _prox_ssl_block_logdensity_and_gradient!(grad, target, u)
    ok || throw(ArgumentError("current HMC block state has non-finite logdensity or gradient."))

    q = copy(u)
    momentum = Vector{Float64}(undef, d)
    _prox_ssl_block_sample_momentum!(momentum, state.inv_metric, rng)
    initial_energy = -current_lp + _prox_ssl_block_kinetic(momentum, state.inv_metric)

    q_prop = copy(q)
    momentum_prop = copy(momentum)
    grad_prop = copy(grad)
    proposed_lp = current_lp
    ok_prop = true
    n_steps = 0
    for _ in 1:state.n_leapfrog
        proposed_lp, ok_prop = _prox_ssl_block_leapfrog!(
            q_prop,
            momentum_prop,
            grad_prop,
            target,
            state.inv_metric,
            state.step_size,
        )
        n_steps += 1
        ok_prop || break
    end

    proposed_energy = -proposed_lp + _prox_ssl_block_kinetic(momentum_prop, state.inv_metric)
    energy_error = proposed_energy - initial_energy
    divergent = !ok_prop || !isfinite(energy_error) || energy_error > state.divergence_threshold
    accept_prob = divergent ? 0.0 : min(1.0, _prox_ssl_block_accept_probability(energy_error))
    accepted = !divergent && rand(rng) < accept_prob
    if accepted
        u .= q_prop
        current_lp = proposed_lp
    end
    _prox_ssl_block_record_transition!(state, accept_prob, divergent)
    return ProxSSLassoBlockTransition(
        accepted,
        accept_prob,
        current_lp,
        energy_error,
        divergent,
        0,
        n_steps,
        state.step_size,
    )
end

struct _ProxSSLassoNUTSTree
    q_minus::Vector{Float64}
    r_minus::Vector{Float64}
    grad_minus::Vector{Float64}
    q_plus::Vector{Float64}
    r_plus::Vector{Float64}
    grad_plus::Vector{Float64}
    q_proposal::Vector{Float64}
    lp_proposal::Float64
    n_valid::Int
    keep_going::Bool
    alpha_sum::Float64
    n_alpha::Int
    divergent::Bool
    max_energy_error::Float64
    n_leapfrog::Int
end

function _prox_ssl_no_u_turn(
    q_minus::AbstractVector,
    q_plus::AbstractVector,
    r_minus::AbstractVector,
    r_plus::AbstractVector,
    inv_metric::AbstractVector,
)
    left = 0.0
    right = 0.0
    @inbounds for i in eachindex(q_minus, q_plus, r_minus, r_plus, inv_metric)
        delta = q_plus[i] - q_minus[i]
        left += delta * inv_metric[i] * r_minus[i]
        right += delta * inv_metric[i] * r_plus[i]
    end
    return left >= 0 && right >= 0
end

function _prox_ssl_build_nuts_tree(
    q::Vector{Float64},
    momentum::Vector{Float64},
    grad::Vector{Float64},
    logu::Float64,
    direction::Int,
    depth::Int,
    step_size::Float64,
    target::AbstractCoxTarget,
    inv_metric::Vector{Float64},
    joint0::Float64,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    if depth == 0
        q1 = copy(q)
        r1 = copy(momentum)
        grad1 = copy(grad)
        lp1, ok = _prox_ssl_block_leapfrog!(
            q1,
            r1,
            grad1,
            target,
            inv_metric,
            direction * step_size,
        )
        joint = lp1 - _prox_ssl_block_kinetic(r1, inv_metric)
        energy_error = joint0 - joint
        divergent = !ok || !isfinite(energy_error) || energy_error > divergence_threshold
        n_valid = !divergent && logu <= joint ? 1 : 0
        keep_going = !divergent && logu - divergence_threshold < joint
        alpha = min(1.0, _prox_ssl_block_accept_probability(energy_error))
        return _ProxSSLassoNUTSTree(
            q1,
            r1,
            grad1,
            q1,
            r1,
            grad1,
            q1,
            lp1,
            n_valid,
            keep_going,
            alpha,
            1,
            divergent,
            energy_error,
            1,
        )
    end

    first = _prox_ssl_build_nuts_tree(
        q,
        momentum,
        grad,
        logu,
        direction,
        depth - 1,
        step_size,
        target,
        inv_metric,
        joint0,
        divergence_threshold,
        rng,
    )
    first.keep_going || return first

    second_start_q = direction == -1 ? first.q_minus : first.q_plus
    second_start_r = direction == -1 ? first.r_minus : first.r_plus
    second_start_grad = direction == -1 ? first.grad_minus : first.grad_plus
    second = _prox_ssl_build_nuts_tree(
        second_start_q,
        second_start_r,
        second_start_grad,
        logu,
        direction,
        depth - 1,
        step_size,
        target,
        inv_metric,
        joint0,
        divergence_threshold,
        rng,
    )

    q_minus = direction == -1 ? second.q_minus : first.q_minus
    r_minus = direction == -1 ? second.r_minus : first.r_minus
    grad_minus = direction == -1 ? second.grad_minus : first.grad_minus
    q_plus = direction == -1 ? first.q_plus : second.q_plus
    r_plus = direction == -1 ? first.r_plus : second.r_plus
    grad_plus = direction == -1 ? first.grad_plus : second.grad_plus
    n_total = first.n_valid + second.n_valid
    choose_second = second.n_valid > 0 && (
        first.n_valid == 0 || rand(rng) < second.n_valid / max(n_total, 1)
    )
    q_proposal = choose_second ? second.q_proposal : first.q_proposal
    lp_proposal = choose_second ? second.lp_proposal : first.lp_proposal
    keep_going = second.keep_going &&
        _prox_ssl_no_u_turn(q_minus, q_plus, r_minus, r_plus, inv_metric)
    return _ProxSSLassoNUTSTree(
        q_minus,
        r_minus,
        grad_minus,
        q_plus,
        r_plus,
        grad_plus,
        q_proposal,
        lp_proposal,
        n_total,
        keep_going,
        first.alpha_sum + second.alpha_sum,
        first.n_alpha + second.n_alpha,
        first.divergent || second.divergent,
        max(first.max_energy_error, second.max_energy_error),
        first.n_leapfrog + second.n_leapfrog,
    )
end

function prox_ssl_block_nuts_step!(
    u::Vector{Float64},
    target::AbstractCoxTarget,
    state::ProxSSLassoBlockNUTSState;
    rng::AbstractRNG,
)
    d = length(u)
    length(state.inv_metric) == d ||
        throw(DimensionMismatch("NUTS block state metric length must match current block."))
    grad0 = Vector{Float64}(undef, d)
    current_lp, ok = _prox_ssl_block_logdensity_and_gradient!(grad0, target, u)
    ok || throw(ArgumentError("current NUTS block state has non-finite logdensity or gradient."))

    momentum0 = Vector{Float64}(undef, d)
    _prox_ssl_block_sample_momentum!(momentum0, state.inv_metric, rng)
    joint0 = current_lp - _prox_ssl_block_kinetic(momentum0, state.inv_metric)
    logu = joint0 - randexp(rng)

    q_minus = copy(u)
    q_plus = copy(u)
    r_minus = copy(momentum0)
    r_plus = copy(momentum0)
    grad_minus = copy(grad0)
    grad_plus = copy(grad0)
    q_proposal = copy(u)
    lp_proposal = current_lp
    n_valid = 1
    keep_going = true
    depth = 0
    alpha_sum = 0.0
    n_alpha = 0
    divergent = false
    max_energy_error = 0.0
    n_leapfrog = 0

    while keep_going && depth < state.max_depth
        direction = rand(rng, Bool) ? 1 : -1
        tree =
            if direction == -1
                _prox_ssl_build_nuts_tree(
                    q_minus,
                    r_minus,
                    grad_minus,
                    logu,
                    direction,
                    depth,
                    state.step_size,
                    target,
                    state.inv_metric,
                    joint0,
                    state.divergence_threshold,
                    rng,
                )
            else
                _prox_ssl_build_nuts_tree(
                    q_plus,
                    r_plus,
                    grad_plus,
                    logu,
                    direction,
                    depth,
                    state.step_size,
                    target,
                    state.inv_metric,
                    joint0,
                    state.divergence_threshold,
                    rng,
                )
            end
        if direction == -1
            q_minus = tree.q_minus
            r_minus = tree.r_minus
            grad_minus = tree.grad_minus
        else
            q_plus = tree.q_plus
            r_plus = tree.r_plus
            grad_plus = tree.grad_plus
        end
        if tree.keep_going && tree.n_valid > 0 && rand(rng) < tree.n_valid / max(n_valid + tree.n_valid, 1)
            q_proposal = tree.q_proposal
            lp_proposal = tree.lp_proposal
        end
        n_valid += tree.n_valid
        keep_going = tree.keep_going &&
            _prox_ssl_no_u_turn(q_minus, q_plus, r_minus, r_plus, state.inv_metric)
        alpha_sum += tree.alpha_sum
        n_alpha += tree.n_alpha
        divergent |= tree.divergent
        max_energy_error = max(max_energy_error, tree.max_energy_error)
        n_leapfrog += tree.n_leapfrog
        depth += 1
    end

    accepted = any(i -> q_proposal[i] != u[i], eachindex(u))
    u .= q_proposal
    accept_prob = n_alpha == 0 ? 0.0 : alpha_sum / n_alpha
    _prox_ssl_block_record_transition!(state, accept_prob, divergent)
    return ProxSSLassoBlockTransition(
        accepted,
        accept_prob,
        lp_proposal,
        max_energy_error,
        divergent,
        depth,
        n_leapfrog,
        state.step_size,
    )
end
