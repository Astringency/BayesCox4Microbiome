function _adaptive_mh(
    logpost::Function,
    init::AbstractVector{<:Real},
    names::Vector{String};
    n_samples::Integer,
    warmup::Integer,
    thin::Integer,
    rng::AbstractRNG,
    initial_step::Union{Real,AbstractVector{<:Real}},
    target_accept::Real,
    adapt_every::Integer,
    min_step::Real,
    max_step::Real,
    center_first::Integer,
    sum_step::Real,
    progress::Bool,
    progress_desc::AbstractString = "MH sampling",
    progress_offset::Integer = 0,
)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive."))
    warmup >= 0 || throw(ArgumentError("warmup must be nonnegative."))
    thin > 0 || throw(ArgumentError("thin must be positive."))
    adapt_every > 0 || throw(ArgumentError("adapt_every must be positive."))

    u = collect(Float64, init)
    d = length(u)
    length(names) == d || throw(DimensionMismatch("names length must match state length."))
    step = initial_step isa Real ? fill(Float64(initial_step), d) : collect(Float64, initial_step)
    length(step) == d || throw(DimensionMismatch("initial_step length must match state length."))
    all(>(0), step) || throw(ArgumentError("initial_step entries must be positive."))
    center_first >= 0 || throw(ArgumentError("center_first must be nonnegative."))
    center_first <= d || throw(ArgumentError("center_first cannot exceed state dimension."))
    sum_step >= 0 || throw(ArgumentError("sum_step must be nonnegative."))

    total_iter = warmup + n_samples * thin
    draws = Matrix{Float64}(undef, n_samples, d)
    lps = Vector{Float64}(undef, n_samples)

    current_lp = logpost(u)
    isfinite(current_lp) || throw(ArgumentError("initial state has non-finite log posterior."))

    mu = copy(u)
    m2 = zeros(d)
    n_adapt = 1
    global_scale = 1.0
    accept_warmup = 0
    accept_sample = 0
    accept_window = 0
    saved = 0
    prog = _progress_meter(progress, total_iter, "MH sampling"; progress_desc = progress_desc, progress_offset = progress_offset)

    for iter in 1:total_iter
        noise = step .* randn(rng, d)
        if center_first > 1
            beta_noise = view(noise, 1:center_first)
            beta_noise .-= mean(beta_noise)
            if sum_step > 0
                beta_noise .+= sum_step * randn(rng) / center_first
            end
        end
        prop = u .+ global_scale .* noise
        prop_lp = logpost(prop)
        accepted = isfinite(prop_lp) && log(rand(rng)) < prop_lp - current_lp
        if accepted
            u = prop
            current_lp = prop_lp
            if iter <= warmup
                accept_warmup += 1
                accept_window += 1
            else
                accept_sample += 1
            end
        end

        if iter <= warmup
            n_adapt += 1
            delta = u .- mu
            mu .+= delta ./ n_adapt
            m2 .+= delta .* (u .- mu)

            if iter % adapt_every == 0 && n_adapt > 20
                empirical_sd = sqrt.(max.(m2 ./ max(n_adapt - 1, 1), eps(Float64)))
                step .= clamp.(2.38 / sqrt(d) .* empirical_sd, min_step, max_step)
                window_rate = accept_window / adapt_every
                global_scale *= exp(clamp(window_rate - target_accept, -0.25, 0.25))
                accept_window = 0
            end
        elseif (iter - warmup) % thin == 0
            saved += 1
            @views draws[saved, :] .= u
            lps[saved] = current_lp
        end
        phase = iter <= warmup ? "warmup $(iter)/$(warmup)" : "sample $(saved)/$(n_samples)"
        _progress_next!(prog; showvalues = [(:phase, phase), (:saved, saved)])
    end
    _progress_finish!(prog)

    sample_iters = n_samples * thin
    warmup_rate = warmup == 0 ? NaN : accept_warmup / warmup
    sample_rate = sample_iters == 0 ? NaN : accept_sample / sample_iters
    return MHSamples(draws, lps, names, sample_rate, warmup_rate, global_scale .* step)
end

function _sample_mh(
    data::CoxPreparedData,
    prior::AbstractCoxPrior;
    n_samples::Integer = 5000,
    warmup::Integer = 2000,
    thin::Integer = 1,
    rng::AbstractRNG = Random.default_rng(),
    initial = nothing,
    initial_step::Union{Real,AbstractVector{<:Real}} = 0.05,
    target_accept::Real = 0.234,
    adapt_every::Integer = 50,
    min_step::Real = 1e-5,
    max_step::Real = 1.0,
    constraint::Symbol = :soft_center,
    prox_sum0_lambda::Real = 1e-3,
    init::Symbol = :zero,
    center_beta_proposal::Bool = true,
    car_weights = nothing,
    use_car::Bool = car_weights !== nothing,
    car_precision::Real = 1.0,
    sample_car_strength::Bool = false,
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_degree_scale::Symbol = :mean_active,
    car_kappa_init::Real = 0.5,
    progress::Bool = true,
    progress_desc::AbstractString = "MH sampling",
    progress_offset::Integer = 0,
)
    p = size(data.z, 2)
    n_events = count(!=(0), data.event)
    effective_car_weights = use_car ? car_weights : nothing
    if sample_car_strength
        _validate_adaptive_car_options(
            effective_car_weights,
            car_strength_prior_a,
            car_strength_prior_b,
            car_kappa_init,
        )
    end
    init =
        if initial !== nothing
            collect(Float64, initial)
        elseif prior isa RegularizedHorseshoePrior
            initial_state(
                prior,
                p,
                n_events;
                n_obs = size(data.z, 1),
                init = init,
                car_weights = effective_car_weights,
                sample_car_strength = sample_car_strength,
                car_strength_prior_a = car_strength_prior_a,
                car_strength_prior_b = car_strength_prior_b,
                car_kappa_init = car_kappa_init,
            )
        else
            initial_state(
                prior,
                p,
                n_events;
                init = init,
                car_weights = effective_car_weights,
                sample_car_strength = prior isa Union{HorseshoePrior,RegularizedHorseshoePrior} ? sample_car_strength : false,
                car_strength_prior_a = car_strength_prior_a,
                car_strength_prior_b = car_strength_prior_b,
                car_kappa_init = car_kappa_init,
            )
    end
    names = _state_names(
        prior,
        p;
        sample_car_strength = prior isa Union{HorseshoePrior,RegularizedHorseshoePrior} ? sample_car_strength : false,
    )
    resolved_constraint = _resolve_constraint(constraint)
    logpost = u -> logposterior(
        u,
        data,
        prior;
        constraint = resolved_constraint,
        prox_sum0_lambda = prox_sum0_lambda,
        car_weights = car_weights,
        use_car = use_car,
        car_precision = car_precision,
        sample_car_strength = sample_car_strength,
        car_strength_prior_a = car_strength_prior_a,
        car_strength_prior_b = car_strength_prior_b,
        car_degree_scale = car_degree_scale,
        car_kappa_init = car_kappa_init,
    )
    return _adaptive_mh(
        logpost,
        init,
        names;
        n_samples = n_samples,
        warmup = warmup,
        thin = thin,
        rng = rng,
        initial_step = initial_step,
        target_accept = target_accept,
        adapt_every = adapt_every,
        min_step = min_step,
        max_step = max_step,
        center_first = center_beta_proposal ? p : 0,
        sum_step = resolved_constraint == :soft_center ? _default_soft_center_scale(p) : 0.0,
        progress = progress,
        progress_desc = progress_desc,
        progress_offset = progress_offset,
    )
end
