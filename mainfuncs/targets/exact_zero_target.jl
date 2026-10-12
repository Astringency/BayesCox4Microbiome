"""
Orthonormal coordinates for `a' * beta = 0`. The Householder construction
avoids choosing a reference taxon. Positive `a` also covers coefficients whose
design columns were standardized (`a = 1 ./ col_scale`).
"""
function _zero_sum_basis(a::AbstractVector)
    length(a) >= 2 || throw(ArgumentError("exact zero sum requires at least two taxa"))
    all(x -> isfinite(x) && x > 0, a) || throw(ArgumentError("constraint weights must be finite and positive"))
    w = Float64.(a) / norm(a)
    h = copy(w)
    h[end] += 1
    H = Matrix{Float64}(I, length(a), length(a) - 1)
    H .-= (2 / dot(h, h)) .* (h * h[1:end-1]')
    return H
end

"""
Exact zero-sum target. Gaussian shrinkage uses its *normalized conditional*
prior on the hyperplane. Prox-SSLASSO uses the existing joint Moreau prior
kernel restricted to that hyperplane, including its original alpha and
indicator factors. These are distinct prior conventions; see README.md.
"""
struct CoxExactZeroTarget{T<:AbstractCoxTarget} <: AbstractCoxTarget
    base::T
    a::Vector{Float64}
    H::Matrix{Float64}
    names::Vector{String}
end

function CoxExactZeroTarget(base::AbstractCoxTarget)
    base.constraint == :none || throw(ArgumentError("exact target requires an unpenalized base target"))
    supported = base isa CoxNoncenteredPosteriorTarget ||
        (base isa CoxPosteriorTarget && base.prior isa Union{HorseshoePrior,RegularizedHorseshoePrior}) ||
        base isa CoxProxSSLassoTarget
    supported || throw(ArgumentError("exact zero sum supports HS/RHS and proximal SSLASSO"))
    p = size(base.data.z, 2)
    a = 1.0 ./ base.data.col_scale
    names = vcat(["contrast[$j]" for j in 1:(p-1)], state_names(base)[p+1:end])
    return CoxExactZeroTarget(base, a, _zero_sum_basis(a), names)
end

state_names(target::CoxExactZeroTarget) = target.names
sample_state_names(target::CoxExactZeroTarget) = sample_state_names(target.base)

function _exact_logscales(base, full::AbstractVector)
    p = size(base.data.z, 2)
    log_tau = full[p+1]
    logs = log_tau .+ full[p+2:2p+1]
    if base.prior isa RegularizedHorseshoePrior
        slab = full[2p+2] / 2
        # Stable harmonic combination; agrees with the existing RHS scales.
        logs = [min(x, slab) - log1p(exp(-2abs(x-slab))) / 2 for x in logs]
    end
    return logs
end

function _exact_householder(target::CoxExactZeroTarget, full::AbstractVector)
    logs = _exact_logscales(target.base, full) .+ log.(target.a)
    v = exp.(logs .- maximum(logs))
    w = v / norm(v)
    h = copy(w)
    h[end] += 1
    return w, h, dot(h, h)
end

function _exact_full_state(target::CoxExactZeroTarget, u::AbstractVector)
    length(u) == dimension(target) || throw(DimensionMismatch("wrong exact state dimension"))
    p = length(target.a)
    full = vcat(zeros(p), u[p:end])
    if target.base isa CoxNoncenteredPosteriorTarget
        _, h, d = _exact_householder(target, full)
        x = vcat(u[1:p-1], 0.0)
        full[1:p] .= x .- (2dot(h, x)/d) .* h
    else
        full[1:p] .= target.H * view(u, 1:p-1)
    end
    return full
end

function _exact_internal_state(target::CoxExactZeroTarget, full::AbstractVector)
    p = length(target.a)
    length(full) == dimension(target.base) || throw(DimensionMismatch("wrong full state dimension"))
    coords = if target.base isa CoxNoncenteredPosteriorTarget
        _, h, d = _exact_householder(target, full)
        x = full[1:p] .- (2dot(h, view(full, 1:p))/d) .* h
        x[1:p-1]
    else
        target.H' * view(full, 1:p)
    end
    return vcat(coords, full[p+1:end])
end

function initial_state(target::CoxExactZeroTarget, rng::AbstractRNG = Random.default_rng())
    base = target.base
    full = base isa CoxPosteriorTarget ? initial_state(base) : initial_state(base, rng)
    return _exact_internal_state(target, full)
end

_prepare_explicit_initial(target::AbstractCoxTarget, initial) = initial
function _prepare_explicit_initial(target::CoxExactZeroTarget, initial)
    initial === nothing && return nothing
    return length(initial) == dimension(target.base) ? _exact_internal_state(target, initial) : initial
end

sample_state(target::CoxExactZeroTarget, u::AbstractVector) =
    sample_state(target.base, _exact_full_state(target, u))

# The integral of the full Gaussian density over the hyperplane is
# norm(a) / sqrt(2pi * a'Da). This correction preserves the hyperprior.
function _exact_gaussian_correction(target::CoxExactZeroTarget, full::AbstractVector)
    logs = 2 .* (_exact_logscales(target.base, full) .+ log.(target.a))
    m = maximum(logs)
    return 0.5 * (_LOG2PI + m + log(sum(exp.(logs .- m)))) - log(norm(target.a))
end

function _exact_scale_gradient!(g, target::CoxExactZeroTarget, full, weights)
    p = length(target.a)
    for j in 1:p
        if target.base.prior isa RegularizedHorseshoePrior
            slab_fraction = _logistic(2(full[p+1] + full[p+1+j]) - full[2p+2])
            raw_fraction = 1 - slab_fraction
            g[p+1] += weights[j] * raw_fraction
            g[p+1+j] += weights[j] * raw_fraction
            g[2p+2] += weights[j] * 0.5 * slab_fraction
        else
            g[p+1] += weights[j]
            g[p+1+j] += weights[j]
        end
    end
    return g
end

function logdensity(target::CoxExactZeroTarget, u::AbstractVector)
    full = _exact_full_state(target, u)
    lp = logdensity(target.base, full)
    if target.base isa CoxPosteriorTarget
        lp += _exact_gaussian_correction(target, full)
    elseif target.base isa CoxNoncenteredPosteriorTarget
        lp += 0.5 * _LOG2PI # p standard normals -> p-1 standard normals
    end
    return lp
end

function logdensity_and_gradient!(grad::AbstractVector{Float64}, target::CoxExactZeroTarget, u::AbstractVector)
    p = length(target.a)
    full = _exact_full_state(target, u)
    g = zeros(length(full))
    lp, _ = logdensity_and_gradient!(g, target.base, full)
    if target.base isa CoxNoncenteredPosteriorTarget
        w, h, d = _exact_householder(target, full)
        x = vcat(u[1:p-1], 0.0)
        t = dot(h, x)
        b = view(g, 1:p)
        hb = dot(h, b)
        grad[1:p-1] .= (b .- (2hb/d) .* h)[1:p-1]
        gh = (-2/d) .* (hb .* x .+ t .* b) .+ (4t*hb/d^2) .* h
        # derivative through w = (a .* sigma) / norm(a .* sigma)
        scale_g = w .* (gh .- dot(w, gh) .* w)
        _exact_scale_gradient!(g, target, full, scale_g)
        lp += 0.5 * _LOG2PI
    else
        grad[1:p-1] .= target.H' * view(g, 1:p)
        if target.base isa CoxPosteriorTarget
            logs = 2 .* (_exact_logscales(target.base, full) .+ log.(target.a))
            weights = exp.(logs .- maximum(logs))
            weights ./= sum(weights)
            _exact_scale_gradient!(g, target, full, weights)
            lp += _exact_gaussian_correction(target, full)
        end
    end
    grad[p:end] .= view(g, p+1:length(g))
    return lp, grad
end

function gradient!(grad::AbstractVector{Float64}, target::CoxExactZeroTarget, u::AbstractVector)
    logdensity_and_gradient!(grad, target, u)
    return grad
end

function sample_logdensity(target::CoxExactZeroTarget, saved::AbstractVector, internal::AbstractVector)
    if target.base isa CoxNoncenteredPosteriorTarget
        # MAP summaries use the constrained centered density, just as existing
        # noncentered samplers report their centered posterior density.
        return sample_logdensity(target.base, saved, _exact_full_state(target, internal)) +
            _exact_gaussian_correction(target, saved)
    end
    return logdensity(target, internal)
end
