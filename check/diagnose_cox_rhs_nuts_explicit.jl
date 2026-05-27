#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "scripts", "analysis_common.jl"))

using AdvancedHMC
using BayesCox4Micro
using LogDensityProblems
using Random
using Statistics

const B = BayesCox4Micro

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))
env_bool(name, default) = lowercase(get(ENV, name, string(default))) in ("1", "true", "yes", "y")
env_symbol(name, default) = Symbol(get(ENV, name, string(default)))

mutable struct CallCounters
    logp_calls::Int
    grad_calls::Int
    logp_bad::Int
    grad_bad::Int
    exceptions::Int
    records::Vector{NamedTuple}
    top_grad_records::Vector{NamedTuple}
end

CallCounters() = CallCounters(0, 0, 0, 0, 0, NamedTuple[], NamedTuple[])

function finite_range(v)
    vals = Float64[x for x in v if isfinite(Float64(x))]
    isempty(vals) && return (NaN, NaN)
    return (minimum(vals), maximum(vals))
end

function state_summary(u, target, err = nothing)
    p = size(target.data.z, 2)
    beta = view(u, 1:p)
    log_tau = Float64(u[p + 1])
    log_lambda = view(u, (p + 2):(2p + 1))
    log_c2 = Float64(u[2p + 2])
    log_raw = log_tau .+ Float64.(log_lambda)
    log_slab = 0.5 * log_c2
    log_beta_scale = similar(log_raw)
    for j in eachindex(log_raw)
        lo = min(log_raw[j], log_slab)
        hi = max(log_raw[j], log_slab)
        log_beta_scale[j] = lo - 0.5 * log1p(exp(2 * (lo - hi)))
    end
    eta = target.data.z * collect(beta)
    scaled_beta_abs = abs.(Float64.(beta)) .* exp.(-log_beta_scale)
    beta_prior_grad_abs = abs.(Float64.(beta)) .* exp.(-2 .* log_beta_scale)
    beta_prior_grad_idx = argmax(beta_prior_grad_abs)
    lp_lik = try
        cox_logpartiallikelihood(target.data, beta)
    catch e
        e
    end
    lp_prior = try
        B._prior_logposterior(u, target.data, target.prior)
    catch e
        e
    end
    lp_total = try
        logdensity(target, u)
    catch e
        e
    end
    return (
        beta_absmax = maximum(abs.(beta)),
        beta_sum = sum(beta),
        eta_min = minimum(eta),
        eta_max = maximum(eta),
        log_tau = log_tau,
        log_lambda_min = minimum(log_lambda),
        log_lambda_max = maximum(log_lambda),
        log_c2 = log_c2,
        log_raw_min = minimum(log_raw),
        log_raw_max = maximum(log_raw),
        log_beta_scale_min = minimum(log_beta_scale),
        log_beta_scale_max = maximum(log_beta_scale),
        scaled_beta_absmax = maximum(scaled_beta_abs),
        beta_prior_grad_absmax = maximum(beta_prior_grad_abs),
        beta_prior_grad_idx = beta_prior_grad_idx,
        beta_at_prior_grad_idx = beta[beta_prior_grad_idx],
        log_beta_scale_at_prior_grad_idx = log_beta_scale[beta_prior_grad_idx],
        lp_lik = lp_lik isa Exception ? sprint(showerror, lp_lik) : Float64(lp_lik),
        lp_prior = lp_prior isa Exception ? sprint(showerror, lp_prior) : Float64(lp_prior),
        lp_total = lp_total isa Exception ? sprint(showerror, lp_total) : Float64(lp_total),
        error = err === nothing ? "" : sprint(showerror, err),
    )
end

function push_top_grad_record!(counters::CallCounters, record)
    push!(counters.top_grad_records, record)
    sort!(counters.top_grad_records; by = x -> x.grad_norm, rev = true)
    if length(counters.top_grad_records) > 8
        pop!(counters.top_grad_records)
    end
    return nothing
end

function push_record!(counters::CallCounters, tag::Symbol, u, target, err = nothing)
    length(counters.records) >= 20 && return nothing
    push!(counters.records, merge((tag = tag,), state_summary(u, target, err)))
    return nothing
end

function stat_value(stat, name::Symbol, default = missing)
    if stat isa NamedTuple
        return haskey(stat, name) ? getfield(stat, name) : default
    end
    return name in propertynames(stat) ? getproperty(stat, name) : default
end

function run_chain(chain::Integer, seed::Integer, sim, data, prior; n_samples, warmup, target_accept, max_depth)
    target = B._make_explicit_nuts_target(
        data,
        prior;
        constraint = :soft_center,
        prox_sum0_lambda = 1e-3,
        prox_l1_lambda = 1e-3,
        alpha_prior_shape = 1.0,
        alpha_prior_rate = 1.0,
        prox_root_tol = 1e-10,
        prox_root_maxiter = 100,
        car_weights = nothing,
        use_car = false,
        car_precision = 1.0,
    )
    rng = MersenneTwister(seed)
    counters = CallCounters()
    init = initial_state(prior, size(data.z, 2), count(!=(0), data.event); n_obs = size(data.z, 1))
    d = dimension(target)

    logp = function (u)
        counters.logp_calls += 1
        try
            lp = logdensity(target, u)
            if !isfinite(lp)
                counters.logp_bad += 1
                push_record!(counters, :logp_nonfinite, u, target)
            end
            return lp
        catch err
            counters.exceptions += 1
            counters.logp_bad += 1
            push_record!(counters, :logp_exception, u, target, err)
            rethrow()
        end
    end

    gradlogp = function (u)
        counters.grad_calls += 1
        try
            lp, grad = LogDensityProblems.logdensity_and_gradient(target, u)
            grad_abs = abs.(grad)
            grad_idx = argmax(grad_abs)
            grad_norm = sqrt(sum(abs2, grad))
            if length(counters.top_grad_records) < 8 || grad_norm > last(counters.top_grad_records).grad_norm
                push_top_grad_record!(counters, merge(
                    (
                        grad_norm = grad_norm,
                        grad_absmax = grad_abs[grad_idx],
                        grad_idx = grad_idx,
                        grad_name = state_names(target)[grad_idx],
                        lp = lp,
                    ),
                    state_summary(u, target),
                ))
            end
            if !isfinite(lp) || any(!isfinite, grad)
                counters.grad_bad += 1
                push_record!(counters, :grad_nonfinite, u, target)
            end
            return lp, grad
        catch err
            counters.exceptions += 1
            counters.grad_bad += 1
            push_record!(counters, :grad_exception, u, target, err)
            rethrow()
        end
    end

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
    samples, stats = AdvancedHMC.sample(
        rng,
        hamiltonian,
        kernel,
        init,
        warmup + n_samples,
        adaptor,
        warmup;
        drop_warmup = true,
        progress = false,
        verbose = false,
    )
    numerical_error = count(s -> stat_value(s, :numerical_error, false) == true, stats)
    divergent = count(s -> stat_value(s, :is_accept, true) == false && stat_value(s, :acceptance_rate, 1.0) == 0.0, stats)
    accept_vals = Float64[Float64(stat_value(s, :acceptance_rate, NaN)) for s in stats if isfinite(Float64(stat_value(s, :acceptance_rate, NaN)))]
    energy_error = Float64[Float64(stat_value(s, :hamiltonian_energy_error, NaN)) for s in stats if isfinite(Float64(stat_value(s, :hamiltonian_energy_error, NaN)))]
    max_energy_error = Float64[Float64(stat_value(s, :max_hamiltonian_energy_error, NaN)) for s in stats if isfinite(Float64(stat_value(s, :max_hamiltonian_energy_error, NaN)))]
    tree_depth = Int[Int(stat_value(s, :tree_depth, 0)) for s in stats]
    n_steps = Int[Int(stat_value(s, :n_steps, 0)) for s in stats]
    draw_mat = reduce(vcat, [reshape(collect(s), 1, :) for s in samples])
    bad_transition_records = NamedTuple[]
    for (iter, stat) in enumerate(stats)
        stat_value(stat, :numerical_error, false) == true || continue
        length(bad_transition_records) >= 8 && break
        push!(bad_transition_records, merge(
            (
                iter = iter,
                is_accept = stat_value(stat, :is_accept, missing),
                acceptance_rate = stat_value(stat, :acceptance_rate, missing),
                hamiltonian_energy_error = stat_value(stat, :hamiltonian_energy_error, missing),
                max_hamiltonian_energy_error = stat_value(stat, :max_hamiltonian_energy_error, missing),
                tree_depth = stat_value(stat, :tree_depth, missing),
                n_steps = stat_value(stat, :n_steps, missing),
                log_density = stat_value(stat, :log_density, missing),
            ),
            state_summary(samples[iter], target),
        ))
    end
    return (
        chain = chain,
        seed = seed,
        initial_step = initial_step,
        stats_len = length(stats),
        stat_fields = isempty(stats) ? Symbol[] : collect(Symbol.(propertynames(first(stats)))),
        numerical_error = numerical_error,
        divergent_proxy = divergent,
        accept_mean = isempty(accept_vals) ? NaN : mean(accept_vals),
        accept_min = isempty(accept_vals) ? NaN : minimum(accept_vals),
        energy_error_min = isempty(energy_error) ? NaN : minimum(energy_error),
        energy_error_max = isempty(energy_error) ? NaN : maximum(energy_error),
        max_energy_error_max = isempty(max_energy_error) ? NaN : maximum(max_energy_error),
        tree_depth_max = isempty(tree_depth) ? 0 : maximum(tree_depth),
        n_steps_max = isempty(n_steps) ? 0 : maximum(n_steps),
        draw_absmax = maximum(abs.(draw_mat)),
        logp_calls = counters.logp_calls,
        grad_calls = counters.grad_calls,
        logp_bad = counters.logp_bad,
        grad_bad = counters.grad_bad,
        exceptions = counters.exceptions,
        records = counters.records,
        top_grad_records = counters.top_grad_records,
        bad_transition_records = bad_transition_records,
    )
end

function main()
    n = env_int("BC4M_N", 200)
    p = env_int("BC4M_P", 150)
    nonzero = env_int("BC4M_NONZERO", 12)
    baseline = env_symbol("BC4M_BASELINE", :weibull)
    seed = env_int("BC4M_SEED", 1)
    chains = env_int("BC4M_CHAINS", 4)
    n_samples = env_int("BC4M_SAMPLES", 100)
    warmup = env_int("BC4M_WARMUP", 100)
    target_accept = env_float("BC4M_TARGET_ACCEPT", 0.8)
    max_depth = env_int("BC4M_MAX_DEPTH", 10)
    standardize = env_bool("BC4M_STANDARDIZE", false)
    pseudocount = env_float("BC4M_PSEUDOCOUNT", 0.0)

    sim = generate_simulation_data(n, p, nonzero; baseline = baseline, seed = seed)
    data = prepare_cox_data(sim.X, sim.time, sim.event; pseudocount = pseudocount, standardize = standardize)
    prior = cox_paper_prior("cox-rhs"; X = sim.X, time = sim.time, event = sim.event, p0 = nonzero, pseudocount = pseudocount)
    println("data=", (
        events = count(!=(0), sim.event),
        zmin = minimum(data.z),
        zmax = maximum(data.z),
        etamin = minimum(sim.eta),
        etamax = maximum(sim.eta),
        tau0 = B._resolved_tau_scale(prior, data),
    ))
    for chain in 1:chains
        result = run_chain(
            chain,
            seed + 1000 * chain,
            sim,
            data,
            prior;
            n_samples = n_samples,
            warmup = warmup,
            target_accept = target_accept,
            max_depth = max_depth,
        )
        println("chain_summary=", merge(result, (
            records = length(result.records),
            top_grad_records = length(result.top_grad_records),
            bad_transition_records = length(result.bad_transition_records),
        )))
        for rec in result.records
            println("bad_state=", rec)
        end
        for rec in result.bad_transition_records
            println("bad_transition=", rec)
        end
        for rec in result.top_grad_records
            println("top_gradient=", rec)
        end
    end
end

main()
