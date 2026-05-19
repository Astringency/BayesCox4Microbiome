#!/usr/bin/env julia

using Pkg

function _activate_repo_project()
    project = abspath(joinpath(@__DIR__, "..", "Project.toml"))
    active = Base.active_project()
    active !== project && Pkg.activate(dirname(project))
end

_activate_repo_project()

include(joinpath(@__DIR__, "analysis_common.jl"))

using Dates

function split_floats(s::AbstractString)
    return parse.(Float64, strip.(split(s, ","; keepempty = false)))
end

function split_float_or_truth_grid(s::AbstractString, truth::Integer)
    out = Float64[]
    for raw in split(s, ","; keepempty = false)
        value = lowercase(strip(raw))
        push!(out, value == "truth" ? Float64(truth) : parse(Float64, value))
    end
    return unique(out)
end

function compact_value(x)
    s = replace(string(x), "." => "p", "-" => "m", "+" => "")
    return replace(s, r"[^A-Za-z0-9pmeE]" => "")
end

function prior_family(name::AbstractString)
    key = lowercase(replace(name, "_" => "-", " " => "-"))
    key in ("hs", "horseshoe", "cox-hs", "coxhs") && return :hs
    key in ("rhs", "regularized-horseshoe", "new-rhs", "newrhs", "est-rhs", "estrhs", "cox-rhs", "coxrhs") &&
        return :rhs
    key in ("lasso", "bayesian-lasso", "prox-lasso") && return :lasso
    key in ("sslasso", "spike-slab", "spike-and-slab", "prox-sslasso") && return :sslasso
    return :other
end

is_cox_hs_prior(name::AbstractString) = lowercase(replace(name, "_" => "-", " " => "-")) in ("cox-hs", "coxhs")
is_prox_l1_prior_name(name::AbstractString) = lowercase(replace(name, "_" => "-", " " => "-")) in ("prox-lasso", "prox-sslasso")

function prox_configs(prior::AbstractString, args)
    is_prox_l1_prior_name(prior) || return [(
        suffix = "",
        kwargs = NamedTuple(),
        params = NamedTuple(),
    )]
    lambdas = split_floats(arg(args, "prox-l1-lambda-grid", string(arg(args, "prox-l1-lambda", 1e-3))))
    alpha_shapes = split_floats(arg(args, "alpha-prior-shape-grid", arg(args, "prox-alpha-prior-shape-grid", "0.5,1,2")))
    alpha_rates = split_floats(arg(args, "alpha-prior-rate-grid", arg(args, "prox-alpha-prior-rate-grid", "0.05,0.1,0.2")))
    out = NamedTuple[]
    for lambda in lambdas, shape in alpha_shapes, rate in alpha_rates
        push!(out, (
            suffix = "_proxlam$(compact_value(lambda))_ashape$(compact_value(shape))_arate$(compact_value(rate))",
            kwargs = (prox_l1_lambda = lambda, alpha_prior_shape = shape, alpha_prior_rate = rate),
            params = (prox_l1_lambda = lambda, alpha_prior_shape = shape, alpha_prior_rate = rate),
        ))
    end
    return out
end

function with_prox_grid(base_configs, prior::AbstractString, args)
    prox = prox_configs(prior, args)
    return [(
        label = base.label * p.suffix,
        kwargs = merge(base.kwargs, p.kwargs),
        params = merge(base.params, p.params),
    ) for base in base_configs for p in prox]
end

function hyperparameter_configs(prior::AbstractString, args, nonzero::Integer)
    family = prior_family(prior)
    configs = NamedTuple[]
    if family == :hs
        p0s = split_float_or_truth_grid(arg(args, "hs-p0-grid", "truth"), nonzero)
        local_scales = split_floats(arg(args, "hs-local-scale-grid", "0.5,1,2"))
        taus = is_cox_hs_prior(prior) ? [1.0] : split_floats(arg(args, "hs-tau-scale-grid", "0.25,0.5,1,2"))
        for p0 in p0s, local_scale in local_scales, tau in taus
            push!(configs, (
                label = "hs_p0$(compact_value(p0))_local$(compact_value(local_scale))_tau$(compact_value(tau))",
                kwargs = (p0 = p0, hs_local_scale = local_scale, hs_tau_scale = tau),
                params = (p0 = p0, hs_local_scale = local_scale, hs_tau_scale = tau),
            ))
        end
    elseif family == :rhs
        p0s = split_float_or_truth_grid(arg(args, "rhs-p0-grid", "truth"), nonzero)
        local_scales = split_floats(arg(args, "rhs-local-scale-grid", "0.5,1,2"))
        dfs = split_floats(arg(args, "rhs-slab-df-grid", arg(args, "rhs-c2-df-grid", "2,4,8")))
        s2s = split_floats(arg(args, "rhs-slab-s2-grid", arg(args, "rhs-c2-s2-grid", "0.25,0.5,1,2")))
        for p0 in p0s, local_scale in local_scales, df in dfs, s2 in s2s
            push!(configs, (
                label = "rhs_p0$(compact_value(p0))_local$(compact_value(local_scale))_df$(compact_value(df))_s2$(compact_value(s2))",
                kwargs = (p0 = p0, rhs_local_scale = local_scale, rhs_slab_df = df, rhs_slab_s2 = s2),
                params = (p0 = p0, rhs_local_scale = local_scale, rhs_slab_df = df, rhs_slab_s2 = s2),
            ))
        end
    elseif family == :lasso
        tau2_shapes = split_floats(arg(args, "lasso-tau2-shape-grid", arg(args, "lasso-local-shape-grid", "0.5,1,2")))
        lambda2_shapes = split_floats(arg(args, "lasso-lambda2-shape-grid", "0.5,1,2"))
        lambda2_rates = split_floats(arg(args, "lasso-lambda2-rate-grid", "0.05,0.1,0.2"))
        for tau2_shape in tau2_shapes, lambda2_shape in lambda2_shapes, lambda2_rate in lambda2_rates
            push!(configs, (
                label = "lasso_tshape$(compact_value(tau2_shape))_lshape$(compact_value(lambda2_shape))_lrate$(compact_value(lambda2_rate))",
                kwargs = (
                    lasso_tau2_shape = tau2_shape,
                    lasso_lambda2_shape = lambda2_shape,
                    lasso_lambda2_rate = lambda2_rate,
                ),
                params = (
                    lasso_tau2_shape = tau2_shape,
                    lasso_lambda2_shape = lambda2_shape,
                    lasso_lambda2_rate = lambda2_rate,
                ),
            ))
        end
    elseif family == :sslasso
        s0s = split_floats(arg(args, "sslasso-s0-grid", arg(args, "sslasso-spike-scale-grid", "0.01,0.02,0.04,0.08")))
        s1s = split_floats(arg(args, "sslasso-s1-grid", arg(args, "sslasso-slab-scale-grid", "0.5,1,2")))
        theta_as = split_floats(arg(args, "sslasso-theta-a-grid", "1"))
        theta_bs = split_floats(arg(args, "sslasso-theta-b-grid", "1,4"))
        for s0 in s0s, s1 in s1s, theta_a in theta_as, theta_b in theta_bs
            s0 < s1 || continue
            push!(configs, (
                label = "ssl_s0$(compact_value(s0))_s1$(compact_value(s1))_ta$(compact_value(theta_a))_tb$(compact_value(theta_b))",
                kwargs = (
                    sslasso_spike_scale = s0,
                    sslasso_slab_scale = s1,
                    sslasso_theta_a = theta_a,
                    sslasso_theta_b = theta_b,
                ),
                params = (
                    sslasso_spike_scale = s0,
                    sslasso_slab_scale = s1,
                    sslasso_theta_a = theta_a,
                    sslasso_theta_b = theta_b,
                ),
            ))
        end
    else
        push!(configs, (label = "default", kwargs = NamedTuple(), params = NamedTuple()))
    end
    configs = with_prox_grid(configs, prior, args)
    max_configs = arg(args, "max-configs-per-prior", 0)
    return max_configs > 0 ? configs[1:min(max_configs, length(configs))] : configs
end

function add_config_columns!(table::DataFrame, config)
    table[!, :config_label] = fill(config.label, nrow(table))
    for (key, value) in pairs(config.params)
        table[!, key] = fill(value, nrow(table))
    end
    return table
end

function append_error_row!(rows::DataFrame; outpath, sim, rep, n, p, nonzero, baseline, prior, config, p0, seed, err)
    row = DataFrame(
        created_at = [string(now())],
        rep = [rep],
        seed = [seed],
        n = [n],
        p = [p],
        nonzero_count = [nonzero],
        baseline = [string(baseline)],
        prior = [prior],
        p0 = [p0 === nothing ? missing : p0],
        estimate = ["error"],
        MSE = [missing],
        event_rate = [mean(sim.event .!= 0)],
        censor_rate = [mean(sim.event .== 0)],
        config_label = [config.label],
        error = [sprint(showerror, err)],
    )
    for (key, value) in pairs(config.params)
        row[!, key] = [value]
    end
    append!(rows, row; cols = :union)
    CSV.write(outpath, rows)
    return rows
end

function main()
    args = parse_cli_args()
    outdir = ensure_dir(arg(args, "outdir", joinpath("results", "prior_hyperparam_tuning")))
    outpath = joinpath(outdir, "hyperparam_tuning_metrics.csv")
    summary_path = joinpath(outdir, "hyperparam_tuning_summary.csv")

    reps = arg(args, "reps", 1)
    ns = split_ints(arg(args, "n", "200"))
    ps = split_ints(arg(args, "p", "50"))
    nonzeros = split_ints(arg(args, "nonzero", "6"))
    baselines = split_symbols(arg(args, "baseline", "weibull"))
    priors = split_strings(arg(args, "priors", "rhs,lasso,sslasso"))
    sampler = Symbol(arg(args, "sampler", "nuts"))
    nuts_backend = Symbol(arg(args, "nuts-backend", "turing"))
    n_samples = arg(args, "samples", 200)
    warmup = arg(args, "warmup", 100)
    chains = arg(args, "chains", 2)
    target_accept = arg(args, "target-accept", 0.8)
    seed0 = arg(args, "seed", 20260516)
    p0_arg = parse_optional_p0(args; default = :truth)
    p0_alpha = arg(args, "p0-alpha", 0.05)
    use_car = arg(args, "use-car", true)
    car_distance = Symbol(arg(args, "car-distance", "bray"))
    pseudocount = arg(args, "pseudocount", 0.0)
    progress = arg(args, "progress", false)
    constraint = Symbol(arg(args, "constraint", "soft_center"))
    soft_center_scale = haskey(args, "soft-center-scale") ? parse(Float64, args["soft-center-scale"]) : nothing
    prox_sum0_lambda = arg(args, "prox-sum0-lambda", 1e-3)
    prox_l1_lambda = arg(args, "prox-l1-lambda", 1e-3)
    alpha_prior_shape = optional_float_arg(args, "alpha-prior-shape", "prox-alpha-prior-shape")
    alpha_prior_rate = optional_float_arg(args, "alpha-prior-rate", "prox-alpha-prior-rate")
    prox_root_tol = arg(args, "prox-root-tol", 1e-10)
    prox_root_maxiter = arg(args, "prox-root-maxiter", 100)

    rows = DataFrame()
    for rep in 1:reps, n in ns, p in ps, nonzero in nonzeros, baseline in baselines
        nonzero <= p || continue
        seed = seed0 + rep - 1
        baseline_args = baseline_parameter_args(args, baseline)
        sim = generate_simulation_data(
            n,
            p,
            nonzero;
            baseline = baseline,
            seed = seed,
            baseline_args...,
        )
        for prior in priors
            p0_requested = p0_arg === :truth ? nonzero : p0_arg
            base_p0 = p0_requested === nothing ? nothing : Float64(p0_requested)
            actual_use_car = use_car && occursin("hs", lowercase(prior))
            for config in hyperparameter_configs(prior, args, nonzero)
                fit_kwargs = merge(
                    (
                        p0 = base_p0,
                        prox_l1_lambda = prox_l1_lambda,
                        alpha_prior_shape = alpha_prior_shape,
                        alpha_prior_rate = alpha_prior_rate,
                    ),
                    config.kwargs,
                )
                try
                    fit = fit_and_evaluate(
                        sim.X,
                        sim.time,
                        sim.event;
                        prior_name = prior,
                        sampler = sampler,
                        nuts_backend = nuts_backend,
                        n_samples = n_samples,
                        warmup = warmup,
                        chains = chains,
                        target_accept = target_accept,
                        seed = seed,
                        use_car = actual_use_car,
                        car_distance = car_distance,
                        p0_alpha = p0_alpha,
                        pseudocount = pseudocount,
                        constraint = constraint,
                        soft_center_scale = soft_center_scale,
                        prox_sum0_lambda = prox_sum0_lambda,
                        prox_root_tol = prox_root_tol,
                        prox_root_maxiter = prox_root_maxiter,
                        progress = progress,
                        fit_kwargs...,
                    )
                    summary = ensure_simulation_summary(fit.summary, sim.beta)
                    metrics = annotate_simulation_metrics(
                        fit.metrics,
                        sim,
                        summary;
                        rep = rep,
                        n = n,
                        p = p,
                        nonzero = nonzero,
                        p0 = fit.p0,
                        baseline = baseline,
                        prior = prior,
                    )
                    metrics[!, :created_at] = fill(string(now()), nrow(metrics))
                    metrics[!, :seed] = fill(seed, nrow(metrics))
                    metrics[!, :event_rate] = fill(mean(sim.event .!= 0), nrow(metrics))
                    metrics[!, :censor_rate] = fill(mean(sim.event .== 0), nrow(metrics))
                    add_config_columns!(metrics, config)
                    append!(rows, metrics; cols = :union)
                    CSV.write(outpath, rows)
                    best = minimum(skipmissing(metrics.MSE))
                    println("prior=$prior config=$(config.label) best_mse=$best")
                catch err
                    append_error_row!(
                        rows;
                        outpath = outpath,
                        sim = sim,
                        rep = rep,
                        n = n,
                        p = p,
                        nonzero = nonzero,
                        baseline = baseline,
                        prior = prior,
                        config = config,
                        p0 = fit_kwargs.p0,
                        seed = seed,
                        err = err,
                    )
                    @warn "Hyperparameter fit failed." prior config=config.label exception=(err, catch_backtrace())
                end
            end
        end
    end

    if nrow(rows) > 0 && :MSE in propertynames(rows)
        ok = rows[rows.estimate .!= "error", :]
        if nrow(ok) > 0
            group_cols = [c for c in [:prior, :config_label, :estimate, :n, :p, :nonzero_count, :baseline] if c in propertynames(ok)]
            summary = combine(
                groupby(ok, group_cols),
                :MSE => mean => :MSE_mean,
                :MSE => std => :MSE_sd,
                :Cindex => mean => :Cindex_mean,
                :LPML => mean => :LPML_mean,
                :ELPD_LOO => mean => :ELPD_LOO_mean,
            )
            sort!(summary, :MSE_mean)
            CSV.write(summary_path, summary)
            show(first(summary, min(20, nrow(summary))); allrows = true, allcols = true)
            println()
        end
    end
    println("Hyperparameter tuning outputs written to $(abspath(outdir))")
end

main()
