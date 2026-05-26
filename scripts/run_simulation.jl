#!/usr/bin/env julia

using Pkg

function _activate_repo_project()
    project = abspath(joinpath(@__DIR__, "..", "Project.toml"))
    active = Base.active_project()
    active !== project && Pkg.activate(dirname(project))
end

_activate_repo_project()

function _include_script(path::AbstractString)
    if get(ENV, "USE_REVISE", "false") in ("1", "true", "yes", "y")
        try
            @eval import Revise
            return Base.invokelatest(Revise.includet, path)
        catch err
            @warn "USE_REVISE was requested but Revise.includet failed; falling back to include." exception=(err, catch_backtrace())
        end
    end
    return include(path)
end

_include_script(joinpath(@__DIR__, "analysis_common.jl"))

function write_simulation_plots(
    run_dir::AbstractString,
    figs_dir::AbstractString,
    summary::DataFrame,
    sim;
    label::AbstractString,
    prior::AbstractString,
    baseline,
    n::Integer,
    p::Integer,
    nonzero::Integer,
)
    coef_paths = [
        joinpath(figs_dir, "coef_$label.pdf"),
        joinpath(figs_dir, "coef_median_$label.pdf"),
        joinpath(figs_dir, "coef_map_$label.pdf"),
    ]
    coefficient_plot(
        summary,
        coef_paths[1];
        truth = sim.beta,
        title = "Simulation $prior $baseline n=$n p=$p nonzero=$nonzero",
        estimate = :mean,
    )
    coefficient_plot(
        summary,
        coef_paths[2];
        truth = sim.beta,
        title = "Simulation $prior median $baseline n=$n p=$p nonzero=$nonzero",
        estimate = :median,
    )
    coefficient_plot(
        summary,
        coef_paths[3];
        truth = sim.beta,
        title = "Simulation $prior MAP $baseline n=$n p=$p nonzero=$nonzero",
        estimate = :map,
    )
    return nothing
end

function _diagnostic_format_list(formats)
    raw =
        formats isa AbstractString ? strip.(split(formats, ",")) :
        formats isa AbstractVector || formats isa Tuple ? strip.(string.(formats)) :
        [strip(string(formats))]
    return [lowercase(fmt) for fmt in raw if !isempty(fmt)]
end

function simulation_diagnostic_outputs_complete(run_dir::AbstractString, formats; make_plots::Bool)
    diag_dir = joinpath(run_dir, "diagnose")
    file_complete(joinpath(diag_dir, "diagnostics_summary.csv")) || return false
    make_plots || return true
    fmt_list = _diagnostic_format_list(formats)
    isempty(fmt_list) && return true
    stems = ("traceplot", "density", "acf")
    return all(file_complete(joinpath(diag_dir, "$stem.$fmt")) for stem in stems for fmt in fmt_list)
end

function write_simulation_diagnostics_from_result(
    run_dir::AbstractString,
    result::CoxSamplerResult,
    sim;
    label::AbstractString,
    diagnose::Bool,
    make_plots::Bool,
    diagnose_formats,
    diagnose_top_k_beta::Integer,
    diagnose_max_lag::Integer,
)
    if diagnose && !simulation_diagnostic_outputs_complete(run_dir, diagnose_formats; make_plots = make_plots)
        nonzero_diagnostic_params = join(("beta[$j]" for j in findall(!iszero, sim.beta)), ",")
        try
            save_diagnostic_outputs(
                result;
                output_dir = joinpath(run_dir, "diagnose"),
                metadata = read_run_metadata(run_dir),
                params = nonzero_diagnostic_params,
                top_k_beta = diagnose_top_k_beta,
                max_lag = diagnose_max_lag,
                formats = diagnose_formats,
                plots = make_plots,
            )
        catch err
            @warn "Diagnostics failed for $label; metrics and samples were kept." exception=(err, catch_backtrace())
        end
    end
    return nothing
end

function write_simulation_diagnostics_from_saved_run(
    run_dir::AbstractString,
    sim;
    label::AbstractString,
    diagnose::Bool,
    make_plots::Bool,
    diagnose_formats,
    diagnose_top_k_beta::Integer,
    diagnose_max_lag::Integer,
)
    if diagnose && !simulation_diagnostic_outputs_complete(run_dir, diagnose_formats; make_plots = make_plots)
        nonzero_diagnostic_params = join(("beta[$j]" for j in findall(!iszero, sim.beta)), ",")
        try
            diagnose_run(
                run_dir;
                output_dir = joinpath(run_dir, "diagnose"),
                params = nonzero_diagnostic_params,
                top_k_beta = diagnose_top_k_beta,
                max_lag = diagnose_max_lag,
                formats = diagnose_formats,
                plots = make_plots,
            )
        catch err
            @warn "Diagnostics failed for $label; metrics and samples were kept." exception=(err, catch_backtrace())
        end
    end
    return nothing
end

function metadata_value_matches(actual, expected)
    actual === missing && return expected === missing
    actual === nothing && return expected === nothing
    if expected isa Number
        actual_number =
            if actual isa Number
                Float64(actual)
            elseif actual isa AbstractString
                try
                    parse(Float64, actual)
                catch
                    return false
                end
            else
                return false
            end
        return isapprox(actual_number, Float64(expected); rtol = 1e-10, atol = 1e-10)
    end
    return string(actual) == string(expected)
end

function simulation_metadata_matches(run_dir::AbstractString, expected::AbstractDict)
    meta = read_run_metadata(run_dir)
    isempty(meta) && return false
    return all(haskey(meta, string(k)) && metadata_value_matches(meta[string(k)], v) for (k, v) in pairs(expected))
end

function main()
    args = parse_cli_args()
    require_known_args!(
        args,
        vcat(
            (
                "outdir",
                "reps",
                "n",
                "p",
                "nonzero",
                "baseline",
                "priors",
                "sampler",
                "nuts-backend",
                "samples",
                "warmup",
                "thin",
                "chains",
                "multichains",
                "target-accept",
                "seed",
                "use-car",
                "progress",
                "car-distance",
                "p0",
                "p0-alpha",
                "constraint",
                "prox-sum0-lambda",
                "prox-l1-lambda",
                "prox-root-tol",
                "prox-root-maxiter",
                "mala-step-size",
                "mala-target-accept",
                "mala-adapt-every",
                "mala-min-step",
                "mala-max-step",
                "diagnose",
                "debug",
                "diagnose-format",
                "diagnose-top-k-beta",
                "diagnose-max-lag",
                "resume",
                "pseudocount",
                "standardize",
                "tie-method",
                "plot-first-rep-only",
                "sample-thin",
                "first-rep-sample-thin",
                "save-wide-csv",
                "save-jld2",
                "alpha-prior-shape",
                "alpha-prior-rate",
            ),
            PRIOR_HYPERPARAMETER_CLI_KEYS,
            BASELINE_PARAMETER_CLI_KEYS,
        ),
    )
    outdir = arg(args, "outdir", joinpath("results", "simulation"))
    reps = arg(args, "reps", 1)
    ns = split_ints(arg(args, "n", "200"))
    ps = split_ints(arg(args, "p", "50"))
    nonzeros = split_ints(arg(args, "nonzero", "6"))
    baselines = split_symbols(arg(args, "baseline", "weibull"))
    priors = split_strings(arg(args, "priors", "hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso"))
    sampler = Symbol(arg(args, "sampler", "nuts"))
    nuts_backend = Symbol(arg(args, "nuts-backend", "turing"))
    n_samples = arg(args, "samples", 500)
    warmup = arg(args, "warmup", 200)
    thin = arg(args, "thin", 1)
    chains = arg(args, "chains", 4)
    multichains = arg(args, "multichains", false)
    target_accept = arg(args, "target-accept", 0.8)
    seed0 = arg(args, "seed", 1)
    use_car = arg(args, "use-car", true)
    progress = arg(args, "progress", true)
    car_distance = Symbol(arg(args, "car-distance", "bray"))
    p0_arg = parse_optional_p0(args; default = nothing)
    p0_alpha = arg(args, "p0-alpha", 0.05)
    constraint = Symbol(arg(args, "constraint", "soft_center"))
    prox_sum0_lambda = arg(args, "prox-sum0-lambda", 1e-3)
    prox_l1_lambda = arg(args, "prox-l1-lambda", 1e-3)
    prox_root_tol = arg(args, "prox-root-tol", 1e-10)
    prox_root_maxiter = arg(args, "prox-root-maxiter", 100)
    mala_step_size = arg(args, "mala-step-size", 1e-4)
    mala_target_accept = arg(args, "mala-target-accept", 0.574)
    mala_adapt_every = arg(args, "mala-adapt-every", 50)
    mala_min_step = arg(args, "mala-min-step", 1e-8)
    mala_max_step = arg(args, "mala-max-step", 1.0)
    diagnose = arg(args, "diagnose", true)
    debug = arg(args, "debug", false)
    diagnose_formats = get(args, "diagnose-format", "pdf")
    diagnose_top_k_beta = arg(args, "diagnose-top-k-beta", 10)
    diagnose_max_lag = arg(args, "diagnose-max-lag", 50)
    resume = debug ? false : arg(args, "resume", true)
    debug && @info "Debug mode forces resume=false."
    pseudocount = arg(args, "pseudocount", 0.0)
    standardize = arg(args, "standardize", false)
    tie_method = parse_tie_method(args)
    plot_first_rep_only = arg(args, "plot-first-rep-only", true)
    sample_thin = max(1, arg(args, "sample-thin", 10))
    first_rep_sample_thin = max(1, arg(args, "first-rep-sample-thin", 1))
    save_wide_csv = arg(args, "save-wide-csv", true)
    save_jld2 = arg(args, "save-jld2", false)
    prior_kwargs = prior_hyperparameter_kwargs(args)
    prior_metadata = prior_hyperparameter_metadata(prior_kwargs)
    alpha_prior_shape = optional_float_arg(args, "alpha-prior-shape")
    alpha_prior_rate = optional_float_arg(args, "alpha-prior-rate")

    tables_dir = ensure_dir(joinpath(outdir, "tables"))
    figs_dir = ensure_dir(joinpath(outdir, "figures"))
    all_metrics = DataFrame()
    all_fd = DataFrame()

    for rep in 1:reps, n in ns, p in ps, nonzero in nonzeros, baseline in baselines
        sim_seed = seed0 + rep - 1
        baseline_args = baseline_parameter_args(args, baseline)
        sim = generate_simulation_data(
            n,
            p,
            nonzero;
            baseline = baseline,
            seed = sim_seed,
            baseline_args...,
        )
        for prior in priors
            p0_requested = p0_arg === :truth ? nonzero : p0_arg
            p0_auto =
                p0_requested === nothing ?
                prior_p0(prior_from_name(
                    prior,
                    sim.X,
                    sim.time,
                    sim.event;
                    p0 = nothing,
                    p0_alpha = p0_alpha,
                    realdata = false,
                    pseudocount = pseudocount,
                )) :
                p0_requested
            p0_value = p0_auto === missing ? nothing : p0_auto
            actual_use_car = use_car && occursin("hs", lowercase(prior))
            label = "rep$(rep)_$(prior)_$(baseline)_n$(n)_p$(p)_nz$(nonzero)"
            run_dir = joinpath(outdir, "runs", label)
            coef_path = joinpath(tables_dir, "coef_$label.csv")
            should_plot = !plot_first_rep_only || rep == 1
            expected_metadata = merge(Dict(
                "baseline_rate" => sim.baseline_rate,
                "baseline_scale" => sim.baseline_scale,
                "baseline_shape" => sim.baseline_shape,
                "censoring_distribution" => string(sim.censoring_distribution),
                "censoring_upper" => sim.censoring_upper,
                "prior" => prior,
                "prior_name" => prior,
                "sampler" => string(sampler),
                "nuts_backend" => string(nuts_backend),
                "samples" => n_samples,
                "n_samples" => n_samples,
                "warmup" => warmup,
                "chains" => chains,
                "p0" => p0_value,
                "constraint" => string(constraint),
                "use_car" => actual_use_car,
                "car_distance" => string(car_distance),
                "alpha_prior_shape" => alpha_prior_shape,
                "alpha_prior_rate" => alpha_prior_rate,
                "pseudocount" => pseudocount,
                "standardize" => standardize,
                "tie_method" => string(tie_method),
                "seed" => sim_seed,
                "git_commit_hash" => BayesCox4Micro._git_commit_hash(),
            ), prior_metadata)
            if resume && simulation_run_recoverable(run_dir) && simulation_metadata_matches(run_dir, expected_metadata)
                try
                    recovered = load_or_recover_simulation_run(
                        run_dir,
                        sim;
                        rep = rep,
                        n = n,
                        p = p,
                        nonzero = nonzero,
                        baseline = baseline,
                        prior = prior,
                        p0 = p0_value,
                    )
                    write_table(coef_path, recovered.summary)
                    append!(all_metrics, recovered.metrics; cols = :union)
                    append!(all_fd, recovered.false_discovery; cols = :union)
                    write_simulation_diagnostics_from_saved_run(
                        run_dir,
                        sim;
                        label = label,
                        diagnose = diagnose,
                        make_plots = should_plot,
                        diagnose_formats = diagnose_formats,
                        diagnose_top_k_beta = diagnose_top_k_beta,
                        diagnose_max_lag = diagnose_max_lag,
                    )
                    if should_plot
                        write_simulation_plots(
                            run_dir,
                            figs_dir,
                            recovered.summary,
                            sim;
                            label = label,
                            prior = prior,
                            baseline = baseline,
                            n = n,
                            p = p,
                            nonzero = nonzero,
                        )
                    end
                    println("[SKIP] $label already has reusable outputs.")
                    continue
                catch err
                    @warn "Could not reuse existing outputs for $label; rerunning this loop." exception=(err, catch_backtrace())
                end
            elseif resume && simulation_run_recoverable(run_dir)
                @info "Existing run uses a different or old censoring setup; rerunning." label run_dir
            end
            fit = fit_and_evaluate(
                sim.X,
                sim.time,
                sim.event;
                prior_name = prior,
                sampler = sampler,
                nuts_backend = nuts_backend,
                n_samples = n_samples,
                warmup = warmup,
                thin = thin,
                chains = chains,
                multichains = multichains,
                target_accept = target_accept,
                seed = sim_seed,
                use_car = actual_use_car,
                car_distance = car_distance,
                pseudocount = pseudocount,
                standardize = standardize,
                tie_method = tie_method,
                p0 = p0_value,
                p0_alpha = p0_alpha,
                constraint = constraint,
                prox_sum0_lambda = prox_sum0_lambda,
                prox_l1_lambda = prox_l1_lambda,
                alpha_prior_shape = alpha_prior_shape,
                alpha_prior_rate = alpha_prior_rate,
                prox_root_tol = prox_root_tol,
                prox_root_maxiter = prox_root_maxiter,
                mala_step_size = mala_step_size,
                mala_target_accept = mala_target_accept,
                mala_adapt_every = mala_adapt_every,
                mala_min_step = mala_min_step,
                mala_max_step = mala_max_step,
                prior_kwargs...,
                progress = progress,
            )
            summary = copy(fit.summary)
            summary.variable .= ["beta[$j]" for j in 1:p]
            summary.true_beta = sim.beta
            summary.nonzero = sim.beta .!= 0
            summary = ensure_simulation_summary(summary, sim.beta)
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
            add_prior_hyperparameter_columns!(metrics, prior_kwargs)
            fd_table = simulation_false_discovery_table(
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
            ensure_dir(run_dir)
            write_table(coef_path, summary)
            write_table(joinpath(run_dir, "posterior_summary.csv"), summary)
            write_table(joinpath(run_dir, "metrics.csv"), metrics)
            write_table(joinpath(run_dir, "false_discovery.csv"), fd_table)
            saved_sample_thin = should_plot ? first_rep_sample_thin : sample_thin
            metadata = merge(Dict(
                "simulation_id" => label,
                "replicate_id" => rep,
                "method_name" => prior,
                "prior" => prior,
                "prior_name" => prior,
                "sampler" => string(sampler),
                "nuts_backend" => string(nuts_backend),
                "baseline" => string(baseline),
                "baseline_rate" => sim.baseline_rate,
                "baseline_scale" => sim.baseline_scale,
                "baseline_shape" => sim.baseline_shape,
                "n" => n,
                "p" => p,
                "nonzero_count" => nonzero,
                "p0" => p0_value,
                "pseudocount" => pseudocount,
                "standardize" => standardize,
                "tie_method" => string(tie_method),
                "censoring_distribution" => string(sim.censoring_distribution),
                "censoring_upper" => sim.censoring_upper,
                "observed_censoring_rate" => mean(sim.event .== 0),
                "observed_event_rate" => mean(sim.event .!= 0),
            ), prior_metadata, Dict(
                "alpha_prior_shape" => alpha_prior_shape,
                "alpha_prior_rate" => alpha_prior_rate,
            ))
            settings = merge(Dict(
                "samples" => n_samples,
                "n_samples" => n_samples,
                "warmup" => warmup,
                "chains" => chains,
                "multichains" => multichains,
                "target_accept" => target_accept,
                "seed" => sim_seed,
                "thin" => thin,
                "saved_sample_thin" => saved_sample_thin,
                "constraint" => string(constraint),
                "prox_sum0_lambda" => prox_sum0_lambda,
                "prox_l1_lambda" => prox_l1_lambda,
                "alpha_prior_shape" => alpha_prior_shape,
                "alpha_prior_rate" => alpha_prior_rate,
                "prox_root_tol" => prox_root_tol,
                "prox_root_maxiter" => prox_root_maxiter,
                "mala_step_size" => mala_step_size,
                "mala_target_accept" => mala_target_accept,
                "mala_adapt_every" => mala_adapt_every,
                "mala_min_step" => mala_min_step,
                "mala_max_step" => mala_max_step,
                "use_car" => actual_use_car,
                "car_distance" => string(car_distance),
                "pseudocount" => pseudocount,
                "standardize" => standardize,
                "tie_method" => string(tie_method),
                "p0_alpha" => p0_alpha,
                "baseline_rate" => sim.baseline_rate,
                "baseline_scale" => sim.baseline_scale,
                "baseline_shape" => sim.baseline_shape,
                "censoring_distribution" => string(sim.censoring_distribution),
                "censoring_upper" => sim.censoring_upper,
            ), prior_metadata)
            save_run_outputs(
                fit.result;
                output_dir = run_dir,
                run_id = label,
                metadata = metadata,
                settings = settings,
                posterior_summary = summary,
                save_wide_csv = save_wide_csv,
                save_jld2 = save_jld2,
                sample_thin = saved_sample_thin,
            )
            mark_simulation_run_complete(run_dir)
            write_simulation_diagnostics_from_result(
                run_dir,
                fit.result,
                sim;
                label = label,
                diagnose = diagnose,
                make_plots = should_plot,
                diagnose_formats = diagnose_formats,
                diagnose_top_k_beta = diagnose_top_k_beta,
                diagnose_max_lag = diagnose_max_lag,
            )
            if should_plot
                write_simulation_plots(
                    run_dir,
                    figs_dir,
                    summary,
                    sim;
                    label = label,
                    prior = prior,
                    baseline = baseline,
                    n = n,
                    p = p,
                    nonzero = nonzero,
                )
            end
            append!(all_metrics, metrics; cols = :union)
            append!(all_fd, fd_table; cols = :union)
        end
    end

    add_elpd_diffs!(
        all_metrics;
        group_cols = [:rep, :n, :p, :nonzero_count, :baseline, :estimate],
    )
    write_table(joinpath(tables_dir, "simulation_metrics_long.csv"), all_metrics)
    write_table(joinpath(tables_dir, "simulation_false_discovery.csv"), all_fd)

    grouped = combine(
        groupby(all_metrics, [:prior, :n, :p, :nonzero_count, :p0, :baseline, :estimate]),
        :MSE => mean => :MSE_mean,
        :MSE => std => :MSE_sd,
        :LPML => mean => :LPML_mean,
        :LPML => std => :LPML_sd,
        :WAIC_pw1 => mean => :WAIC_pw1_mean,
        :WAIC_pw1 => std => :WAIC_pw1_sd,
        :ELPD_LOO => mean => :ELPD_LOO_mean,
        :ELPD_LOO => std => :ELPD_LOO_sd,
        :ELPD_Diff => mean => :ELPD_Diff_mean,
        :ELPD_Diff => std => :ELPD_Diff_sd,
        :PLOO => mean => :PLOO_mean,
        :PLOO => std => :PLOO_sd,
        :LOOIC => mean => :LOOIC_mean,
        :LOOIC => std => :LOOIC_sd,
        :ParetoK_max => maximum => :ParetoK_max,
        :Cindex => mean => :Cindex_mean,
        :Cindex => std => :Cindex_sd,
    )
    write_table(joinpath(tables_dir, "simulation_metrics_summary.csv"), grouped)

    if nrow(all_metrics) > 0
        metrics_plot(all_metrics, joinpath(figs_dir, "simulation_metrics.pdf"); title = "Simulation metrics")
    end
    println("Simulation outputs written to $(abspath(outdir))")
end

main()
