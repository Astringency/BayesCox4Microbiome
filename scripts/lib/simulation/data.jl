# Main experiment and S1–S4 data generation; preserves baseline RNG order.
const DEFAULT_BETA0 = [1.0, -1.0]
const DEFAULT_BETA6 = [2.08, 1.31, -1.39, -1.86, 2.12, -2.26]
const DEFAULT_BETA12 = [2.08, -1.41, -1.39, -1.15, 2.12, 2.51, 1.31, -1.95, -1.86, 1.93, -1.34, -0.85]

const DEFAULT_BASELINE_SETTINGS = Dict(
    :exponential => (rate = 1 / 14, shape = nothing, censoring_upper = 64.0),
    :weibull => (rate = 1 / 144, shape = 2.0, censoring_upper = 35.0),
    :gompertz => (rate = 1 / 25, shape = 0.1, censoring_upper = 35.0),
)

function baseline_default_settings(baseline::Symbol)
    haskey(DEFAULT_BASELINE_SETTINGS, baseline) ||
        throw(ArgumentError("baseline must be :weibull, :exponential, or :gompertz."))
    return DEFAULT_BASELINE_SETTINGS[baseline]
end

function baseline_parameters(baseline::Symbol; rate = nothing, scale = nothing, shape = nothing)
    defaults = baseline_default_settings(baseline)
    if rate !== nothing && scale !== nothing
        throw(ArgumentError("pass either baseline rate or reciprocal-rate scale, not both."))
    end
    rate_value =
        rate !== nothing ? Float64(rate) :
        scale !== nothing ? inv(Float64(scale)) :
        defaults.rate
    rate_value > 0 || throw(ArgumentError("baseline rate must be positive."))

    if baseline == :exponential
        shape === nothing || throw(ArgumentError("exponential baseline does not use a shape parameter."))
        return (rate = rate_value, shape = nothing)
    end

    shape_value = shape === nothing ? defaults.shape : Float64(shape)
    shape_value > 0 || throw(ArgumentError("baseline shape must be positive."))
    return (rate = rate_value, shape = shape_value)
end

baseline_censoring_upper(baseline::Symbol) = baseline_default_settings(baseline).censoring_upper

function baseline_parameter_args(args, baseline::Symbol)
    prefix = string(baseline)
    rate = optional_float_arg(args, "$prefix-rate", "$prefix-lambda", "baseline-rate", "baseline-lambda")
    scale = optional_float_arg(args, "$prefix-scale", "baseline-scale")
    shape =
        baseline == :weibull ?
        optional_float_arg(args, "weibull-shape", "weibull-nu", "baseline-shape", "baseline-nu") :
        baseline == :gompertz ?
        optional_float_arg(args, "gompertz-shape", "gompertz-alpha", "baseline-shape", "baseline-alpha") :
        optional_float_arg(args, "exponential-shape", "baseline-shape")
    return (baseline_rate = rate, baseline_scale = scale, baseline_shape = shape)
end

function true_beta(p::Integer, nonzero::Integer)
    values =
        nonzero == 6 ? DEFAULT_BETA6 :
        nonzero == 12 ? DEFAULT_BETA12 :
        nonzero % 2 == 0 ? vcat(
            fill(DEFAULT_BETA0[1], nonzero ÷ 2),
            fill(DEFAULT_BETA0[2], nonzero ÷ 2),
        ) :
        throw(ArgumentError("nonzero must be 6/12 or a even number."))

    paper_idx = nonzero in (6, 12) ? collect(18:2:(16 + 2nonzero)) : Int[]
    idx =
        !isempty(paper_idx) && maximum(paper_idx) <= p ? paper_idx :
        nonzero <= p ? collect(1:nonzero) :
        throw(ArgumentError("p=$p is too small for nonzero=$nonzero."))
    beta = zeros(Float64, p)
    beta[idx] .= values
    return beta, idx
end

function paper_covariance(p::Integer, index::AbstractVector{<:Integer})
    p > 0 || throw(ArgumentError("p must be positive."))
    idx = collect(Int, index)
    all(i -> 1 <= i <= p, idx) ||
        throw(ArgumentError("covariance index entries must be between 1 and p."))
    length(unique(idx)) == length(idx) ||
        throw(ArgumentError("covariance index entries must be unique."))

    sigma = Matrix{Float64}(I, p, p)
    for a in 1:(length(idx) - 1)
        i = idx[a]
        for b in (a + 1):length(idx)
            j = idx[b]
            sigma[i, j] = 0.75 - 0.015 * abs(i - j)
            sigma[j, i] = sigma[i, j]
        end
    end
    return Symmetric(sigma)
end

function softmax_rows(U::AbstractMatrix)
    X = similar(U, Float64)
    for i in axes(U, 1)
        row = view(U, i, :)
        m = maximum(row)
        denom = sum(exp.(row .- m))
        @views X[i, :] .= exp.(row .- m) ./ denom
    end
    return X
end

function baseline_event_time(
    rng::AbstractRNG,
    eta::Real,
    baseline::Symbol;
    rate = nothing,
    scale = nothing,
    shape = nothing,
)
    params = baseline_parameters(baseline; rate = rate, scale = scale, shape = shape)
    target = -log(rand(rng)) / exp(eta)
    if baseline == :exponential
        return target / params.rate
    elseif baseline == :weibull
        return (target / params.rate)^(1 / params.shape)
    elseif baseline == :gompertz
        return log1p(params.shape * target / params.rate) / params.shape
    else
        throw(ArgumentError("baseline must be :weibull, :exponential, or :gompertz."))
    end
end

function simulation_event_process(
    rng::AbstractRNG,
    n::Integer,
    p::Integer,
    nonzero::Integer;
    baseline::Symbol = :weibull,
    baseline_rate = nothing,
    baseline_scale = nothing,
    baseline_shape = nothing,
    scenario = :baseline,
    weak_multiplier = 0.35,
    null_correlation = 0.6,
)
    scenario = resolve_simulation_scenario(scenario)
    beta, idx = true_beta(p, nonzero)
    if scenario == :weak
        0 < weak_multiplier <= 1 || throw(ArgumentError("weak multiplier must be in (0,1]"))
        beta .*= weak_multiplier
    end
    latent = scenario_latent_parameters(p, idx, scenario; null_correlation)
    U = rand(rng, MvNormal(latent.mu, latent.covariance), n)'
    X = softmax_rows(U)
    eta = log.(X) * beta
    params = baseline_parameters(baseline; rate = baseline_rate, scale = baseline_scale, shape = baseline_shape)
    event_time = [baseline_event_time(rng, eta[i], baseline; rate = params.rate, shape = params.shape) for i in 1:n]
    return (X = X, beta = beta, nonzero_index = idx, eta = eta, event_time = Float64.(event_time),
        scenario = scenario, paired_nulls = latent.paired_nulls, latent_covariance = latent.covariance)
end

function random_censoring_times(
    rng::AbstractRNG,
    n::Integer;
    upper::Real,
)
    upper_value = Float64(upper)
    upper_value > 0 || throw(ArgumentError("censoring upper bound must be positive."))
    return upper_value .* rand(rng, n)
end

function apply_censoring(event_time::AbstractVector{<:Real}, censor_time::AbstractVector{<:Real})
    length(event_time) == length(censor_time) ||
        throw(DimensionMismatch("event_time and censor_time must have the same length."))
    event_time_f = Float64.(event_time)
    censor_time_f = Float64.(censor_time)
    time = min.(event_time_f, censor_time_f)
    event = Int.(event_time_f .<= censor_time_f)
    if count(!=(0), event) == 0
        idx = argmin(event_time_f)
        event[idx] = 1
        time[idx] = event_time_f[idx]
        censor_time_f[idx] = event_time_f[idx]
    end
    return (time = Float64.(time), event = event, censor_time = censor_time_f)
end

function uniform_censoring_rate(event_time::AbstractVector{<:Real}, upper::Real)
    upper_value = Float64(upper)
    upper_value > 0 || throw(ArgumentError("uniform censoring upper bound must be positive."))
    return mean(min.(Float64.(event_time) ./ upper_value, 1.0))
end

function calibrate_uniform_censoring_upper(
    event_time::AbstractVector{<:Real},
    target_censoring_rate::Real;
    tol::Real = 1e-4,
    maxiter::Integer = 100,
)
    target = Float64(target_censoring_rate)
    0 < target < 1 || throw(ArgumentError("target_censoring_rate must be in (0, 1)."))
    finite_times = [Float64(t) for t in event_time if isfinite(Float64(t)) && Float64(t) > 0]
    isempty(finite_times) && throw(ArgumentError("event_time must contain positive finite values."))

    lo = eps(Float64)
    hi = maximum(finite_times)
    while uniform_censoring_rate(finite_times, hi) > target
        hi *= 2
    end

    mid = hi
    for _ in 1:maxiter
        mid = (lo + hi) / 2
        rate = uniform_censoring_rate(finite_times, mid)
        abs(rate - target) <= tol && return mid
        if rate > target
            lo = mid
        else
            hi = mid
        end
    end
    return mid
end

function generate_simulation_data(
    n::Integer,
    p::Integer,
    nonzero::Integer;
    baseline::Symbol = :weibull,
    seed::Integer = 1,
    baseline_rate = nothing,
    baseline_scale = nothing,
    baseline_shape = nothing,
    rng::AbstractRNG = MersenneTwister(seed),
    scenario = :baseline,
    weak_multiplier = 0.35,
    null_correlation = 0.6,
    depth_median = 5000.0,
    depth_log_sd = 0.5,
)
    scenario = resolve_simulation_scenario(scenario)
    params = baseline_parameters(baseline; rate = baseline_rate, scale = baseline_scale, shape = baseline_shape)
    upper = baseline_censoring_upper(baseline)
    process = simulation_event_process(
        rng,
        n,
        p,
        nonzero;
        baseline = baseline,
        baseline_rate = params.rate,
        baseline_shape = params.shape,
        scenario = scenario, weak_multiplier = weak_multiplier, null_correlation = null_correlation,
    )
    censor_time = random_censoring_times(rng, n; upper = upper)
    observed = apply_censoring(process.event_time, censor_time)
    measurement = scenario == :counts ? sequencing_observations(rng, process.X; depth_median, depth_log_sd) :
        (X = process.X, sequencing_counts = nothing, sequencing_depths = nothing, observed_zero_fraction = 0.0)
    return (
        X = measurement.X,
        latent_X = process.X,
        sequencing_counts = measurement.sequencing_counts,
        sequencing_depths = measurement.sequencing_depths,
        observed_zero_fraction = measurement.observed_zero_fraction,
        scenario = scenario,
        paired_nulls = process.paired_nulls,
        latent_covariance = process.latent_covariance,
        time = observed.time,
        event = observed.event,
        beta = process.beta,
        nonzero_index = process.nonzero_index,
        eta = process.eta,
        event_time = process.event_time,
        censor_time = observed.censor_time,
        baseline_rate = params.rate,
        baseline_scale = inv(params.rate),
        baseline_shape = params.shape,
        censoring_distribution = :uniform,
        censoring_upper = upper,
    )
end

"""Independent n=1000 test cohort, shared by methods within a setting/replicate."""
function generate_simulation_test_data(n::Integer, p::Integer, nonzero::Integer;
    seed::Integer = 1, baseline::Symbol = :weibull, kwargs...)
    baseline_id = findfirst(==(baseline), (:weibull, :exponential, :gompertz))
    baseline_id === nothing && throw(ArgumentError("unsupported baseline $baseline"))
    # A separate RNG namespace avoids overlap with any scalar training seed.
    rng = MersenneTwister(UInt32[0x54455354, mod(seed, 2^32), n, p, nonzero, baseline_id])
    sim = generate_simulation_data(1000, p, nonzero; baseline, rng, kwargs...)
    suffix = sim.scenario == :baseline ? "" : ":$(sim.scenario)"
    return merge(sim, (test_seed = "test-v1:$seed:$n:$p:$nonzero:$baseline$suffix",))
end

