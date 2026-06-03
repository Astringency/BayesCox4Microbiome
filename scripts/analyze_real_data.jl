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
_include_script(joinpath(@__DIR__, "tune_hyperparams_cindex.jl"))

function _split_optional_strings(value::AbstractString)
    raw = strip(value)
    isempty(raw) && return String[]
    lowercase(raw) in ("none", "nothing", "missing") && return String[]
    return split_strings(raw)
end

function detect_component_columns(
    df::DataFrame,
    time_col::Symbol,
    status_col::Symbol,
    exclude_cols;
    comp_cols = String[],
)
    if !isempty(comp_cols)
        requested = Symbol.(comp_cols)
        missing_cols = [c for c in requested if !(c in propertynames(df))]
        isempty(missing_cols) ||
            throw(ArgumentError("requested component column(s) not found: $(join(string.(missing_cols), ", "))"))
        return string.(requested)
    end
    excluded = Set(vcat([time_col, status_col], Symbol.(exclude_cols)))
    return [
        c for c in names(df)
        if !(Symbol(c) in excluded)
    ]
end

function matrix_without_missing(df::DataFrame, cols)
    nonnumeric_cols = [c for c in cols if !(eltype(df[!, c]) <: Union{Missing,Real})]
    isempty(nonnumeric_cols) ||
        throw(ArgumentError(
            "component columns must be numeric; nonnumeric column(s): $(join(nonnumeric_cols, ", ")). " *
            "Add them to --exclude-cols or pass an explicit --comp-cols list.",
        ))
    mat = Matrix{Float64}(coalesce.(Matrix(df[:, cols]), 0.0))
    all(>=(0), mat) || throw(ArgumentError("component columns must be nonnegative."))
    return mat
end

function drop_missing_survival_rows(df::DataFrame, time_col::Symbol, status_col::Symbol)
    time_col in propertynames(df) || throw(ArgumentError("time column $time_col was not found."))
    status_col in propertynames(df) || throw(ArgumentError("status column $status_col was not found."))
    keep = .!ismissing.(df[!, time_col]) .& .!ismissing.(df[!, status_col])
    dropped = count(.!keep)
    return df[keep, :], dropped
end

function realdata_cli_valid_keys()
    return vcat(
        (
            "input",
            "outdir",
            "time-col",
            "status-col",
            "comp-cols",
            "exclude-cols",
            "priors",
        ),
        COMMON_ANALYSIS_CLI_KEYS,
    )
end

function main()
    args = parse_cli_args()
    require_known_args!(args, realdata_cli_valid_keys())
    input = arg(args, "input", joinpath("data", "T1D.csv"))
    outdir = arg(args, "outdir", joinpath("results", "realdata", "T1D"))
    time_col = Symbol(arg(args, "time-col", "time"))
    status_col = Symbol(arg(args, "status-col", "status"))
    comp_cols_arg = _split_optional_strings(arg(args, "comp-cols", ""))
    exclude_cols = _split_optional_strings(arg(args, "exclude-cols", "Sex,Antibiotics"))
    priors = split_strings(arg(args, "priors", "hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso"))
    sampler_kwargs = sampler_hyperparameter_kwargs(args)
    car_kwargs = car_hyperparameter_kwargs(args)
    model_kwargs = cox_model_cli_options(args; default_pseudocount = 1e-6, default_standardize = false)
    diagnostic_kwargs = diagnostic_cli_options(args)
    tuning_kwargs = tuning_cli_options(args)
    prior_kwargs = prior_hyperparameter_kwargs(args)
    prior_metadata = prior_hyperparameter_metadata(prior_kwargs)
    sampler = sampler_kwargs.sampler
    nuts_backend = sampler_kwargs.nuts_backend
    n_samples = sampler_kwargs.n_samples
    warmup = sampler_kwargs.warmup
    thin = sampler_kwargs.thin
    chains = sampler_kwargs.chains
    multichains = sampler_kwargs.multichains
    target_accept = sampler_kwargs.target_accept
    max_depth = sampler_kwargs.max_depth
    hmc_n_leapfrog = sampler_kwargs.hmc_n_leapfrog
    hmc_step_size = sampler_kwargs.hmc_step_size
    noncentered = sampler_kwargs.noncentered
    seed = sampler_kwargs.seed
    progress = sampler_kwargs.progress
    use_car = car_kwargs.use_car
    car_distance = car_kwargs.car_distance
    car_weight_quantile = car_kwargs.car_weight_quantile
    car_precision = car_kwargs.car_precision
    sample_car_strength = car_kwargs.sample_car_strength
    car_strength_prior_a = car_kwargs.car_strength_prior_a
    car_strength_prior_b = car_kwargs.car_strength_prior_b
    car_degree_scale = car_kwargs.car_degree_scale
    car_kappa_init = car_kwargs.car_kappa_init
    p0_arg = model_kwargs.p0_arg
    p0_arg === :truth && throw(ArgumentError("--p0 truth is only valid for simulation."))
    p0_alpha = model_kwargs.p0_alpha
    constraint = model_kwargs.constraint
    prox_sum0_lambda = model_kwargs.prox_sum0_lambda
    pseudocount = model_kwargs.pseudocount
    standardize = model_kwargs.standardize
    tie_method = model_kwargs.tie_method
    diagnose = diagnostic_kwargs.diagnose
    tune_priorparams = tuning_kwargs.tune_priorparams
    tune_algparams = tuning_kwargs.tune_algparams
    cv_folds = tuning_kwargs.cv_folds
    tune_seed = tuning_kwargs.tune_seed
    tune_run_best = tuning_kwargs.tune_run_best
    diagnose_formats = diagnostic_kwargs.diagnose_formats
    diagnose_top_k_beta = diagnostic_kwargs.diagnose_top_k_beta
    diagnose_max_lag = diagnostic_kwargs.diagnose_max_lag
    prox_l1_lambda = prior_kwargs.prox_l1_lambda
    alpha_prior_shape = prior_kwargs.alpha_prior_shape# == nothing ? size(X, 2) : prior_kwargs.alpha_prior_shape + size(X, 2)
    alpha_prior_rate = prior_kwargs.alpha_prior_rate
    prox_root_tol = prior_kwargs.prox_root_tol
    prox_root_maxiter = prior_kwargs.prox_root_maxiter
    mala_step_size = sampler_kwargs.mala_step_size
    mala_target_accept = sampler_kwargs.mala_target_accept
    mala_adapt_every = sampler_kwargs.mala_adapt_every
    mala_min_step = sampler_kwargs.mala_min_step
    mala_max_step = sampler_kwargs.mala_max_step
    mala_precondition = sampler_kwargs.mala_precondition
    mala_precondition_mode = sampler_kwargs.mala_precondition_mode
    mala_precondition_diag = sampler_kwargs.mala_precondition_diag
    mala_metric_ridge = sampler_kwargs.mala_metric_ridge
    mala_metric_max_cholesky_tries = sampler_kwargs.mala_metric_max_cholesky_tries
    mala_stepsize_schedule = sampler_kwargs.mala_stepsize_schedule
    mala_sgld_c0 = sampler_kwargs.mala_sgld_c0
    mala_sgld_target_accept = sampler_kwargs.mala_sgld_target_accept
    mala_sgld_accept_low = sampler_kwargs.mala_sgld_accept_low
    mala_sgld_accept_high = sampler_kwargs.mala_sgld_accept_high
    mala_sgld_adapt_c0 = sampler_kwargs.mala_sgld_adapt_c0
    mala_sgld_adapt_every = sampler_kwargs.mala_sgld_adapt_every
    mala_sgld_c0_min = sampler_kwargs.mala_sgld_c0_min
    mala_sgld_c0_max = sampler_kwargs.mala_sgld_c0_max
    prox_block_step_size = sampler_kwargs.prox_block_step_size
    prox_block_min_step_size = sampler_kwargs.prox_block_min_step_size
    prox_block_max_step_size = sampler_kwargs.prox_block_max_step_size
    prox_block_adapt_every = sampler_kwargs.prox_block_adapt_every
    prox_block_divergence_threshold = sampler_kwargs.prox_block_divergence_threshold
    prox_block_metric = sampler_kwargs.prox_block_metric

    tables_dir = ensure_dir(joinpath(outdir, "tables"))
    figs_dir = ensure_dir(joinpath(outdir, "figures"))

    df = CSV.read(input, DataFrame)
    df, dropped_survival_rows = drop_missing_survival_rows(df, time_col, status_col)
    dropped_survival_rows > 0 &&
        @info "Dropped rows with missing survival time/status." dropped_survival_rows time_col status_col
    component_cols = detect_component_columns(
        df,
        time_col,
        status_col,
        exclude_cols;
        comp_cols = comp_cols_arg,
    )
    isempty(component_cols) && throw(ArgumentError("no component columns detected."))
    X = matrix_without_missing(df, component_cols)
    time = Float64.(df[!, time_col])
    event = Int.(df[!, status_col])
    if p0_arg === nothing
        screen_table, screen = univariate_screen_table(X, time, event; names = component_cols, alpha = p0_alpha)
        screen_table.p0 .= screen.p0
        screen_table.alpha .= screen.alpha
        write_table(joinpath(tables_dir, "univariate_cox_screen.csv"), screen_table)
    end

    if alpha_prior_shape !== nothing
        alpha_prior_shape += size(X, 2)
    end

    if tune_priorparams || tune_algparams
        tune_hyperparams_cindex(
            X,
            time,
            event;
            priors = priors,
            base_sampler_kwargs = sampler_kwargs,
            base_prior_kwargs = prior_kwargs,
            base_car_kwargs = car_kwargs,
            base_model_kwargs = (
                p0 = p0_arg,
                p0_alpha = p0_alpha,
                constraint = constraint,
                prox_sum0_lambda = prox_sum0_lambda,
                pseudocount = pseudocount,
                standardize = standardize,
                tie_method = tie_method,
            ),
            outdir = outdir,
            mode = :realdata,
            dataset_id = splitext(basename(input))[1],
            tune_priorparams = tune_priorparams,
            tune_algparams = tune_algparams,
            cv_folds = cv_folds,
            seed = tune_seed,
            realdata = true,
            simulation_metadata = Dict{String,Any}(
                "input" => input,
                "dataset_name" => basename(input),
                "n" => size(X, 1),
                "p" => size(X, 2),
                "events" => count(!=(0), event),
                "event_rate" => mean(event .!= 0),
                "censoring_rate" => mean(event .== 0),
                "seed" => seed,
                "component_cols" => join(component_cols, ","),
                "exclude_cols" => join(exclude_cols, ","),
                "dropped_survival_rows" => dropped_survival_rows,
            ),
            tune_run_best = tune_run_best,
        )
    end

    all_metrics = DataFrame()
    for prior in priors
        if prior in ("lasso", "sslasso", "prox-lasso", "prox-sslasso")
            noncentered = false
        end
        actual_use_car = use_car && occursin("hs", lowercase(prior))
        actual_sample_car_strength = actual_use_car && sample_car_strength
        fit = fit_and_evaluate(
            X,
            time,
            event;
            prior_name = prior,
            sampler = sampler,
            nuts_backend = nuts_backend,
            n_samples = n_samples,
            warmup = warmup,
            thin = thin,
            chains = chains,
            multichains = multichains,
            target_accept = target_accept,
            max_depth = max_depth,
            hmc_n_leapfrog = hmc_n_leapfrog,
            hmc_step_size = hmc_step_size,
            noncentered = noncentered,
            seed = seed,
            use_car = actual_use_car,
            car_distance = car_distance,
            car_weight_quantile = car_weight_quantile,
            car_precision = car_precision,
            sample_car_strength = actual_sample_car_strength,
            car_strength_prior_a = car_strength_prior_a,
            car_strength_prior_b = car_strength_prior_b,
            car_degree_scale = car_degree_scale,
            car_kappa_init = car_kappa_init,
            pseudocount = pseudocount,
            standardize = standardize,
            tie_method = tie_method,
            p0 = p0_arg,
            p0_alpha = p0_alpha,
            constraint = constraint,
            prox_sum0_lambda = prox_sum0_lambda,
            mala_step_size = mala_step_size,
            mala_target_accept = mala_target_accept,
            mala_adapt_every = mala_adapt_every,
            mala_min_step = mala_min_step,
            mala_max_step = mala_max_step,
            mala_precondition = mala_precondition,
            mala_precondition_mode = mala_precondition_mode,
            mala_precondition_diag = mala_precondition_diag,
            mala_metric_ridge = mala_metric_ridge,
            mala_metric_max_cholesky_tries = mala_metric_max_cholesky_tries,
            mala_stepsize_schedule = mala_stepsize_schedule,
            mala_sgld_c0 = mala_sgld_c0,
            mala_sgld_target_accept = mala_sgld_target_accept,
            mala_sgld_accept_low = mala_sgld_accept_low,
            mala_sgld_accept_high = mala_sgld_accept_high,
            mala_sgld_adapt_c0 = mala_sgld_adapt_c0,
            mala_sgld_adapt_every = mala_sgld_adapt_every,
            mala_sgld_c0_min = mala_sgld_c0_min,
            mala_sgld_c0_max = mala_sgld_c0_max,
            prox_block_step_size = prox_block_step_size,
            prox_block_min_step_size = prox_block_min_step_size,
            prox_block_max_step_size = prox_block_max_step_size,
            prox_block_adapt_every = prox_block_adapt_every,
            prox_block_divergence_threshold = prox_block_divergence_threshold,
            prox_block_metric = prox_block_metric,
            prior_kwargs...,
            realdata = true,
            progress = progress,
        )
        summary = copy(fit.summary)
        summary.variable = string.(component_cols)
        summary.selected = (summary.lower .* summary.upper) .> 0
        write_table(joinpath(tables_dir, "coef_$prior.csv"), summary)
        run_id = "realdata_$(prior)"
        run_dir = joinpath(outdir, "runs", run_id)
        save_run_outputs(
            fit.result;
            output_dir = run_dir,
            run_id = run_id,
            metadata = merge(Dict(
                "dataset_name" => basename(input),
                "method_name" => prior,
                "prior_name" => prior,
                "sampler" => string(sampler),
                "nuts_backend" => string(nuts_backend),
                "samples" => n_samples,
                "n_samples" => n_samples,
                "warmup" => warmup,
                "thin" => thin,
                "chains" => chains,
                "multichains" => multichains,
                "target_accept" => target_accept,
                "max_depth" => max_depth,
                "hmc_n_leapfrog" => hmc_n_leapfrog,
                "hmc_step_size" => hmc_step_size,
                "noncentered" => noncentered,
                "p0" => fit.p0,
                "pseudocount" => pseudocount,
                "standardize" => standardize,
                "tie_method" => string(tie_method),
                "constraint" => string(constraint),
                "prox_sum0_lambda" => prox_sum0_lambda,
                "mala_step_size" => mala_step_size,
                "mala_target_accept" => mala_target_accept,
                "mala_adapt_every" => mala_adapt_every,
                "mala_min_step" => mala_min_step,
                "mala_max_step" => mala_max_step,
                "mala_precondition" => mala_precondition,
                "mala_precondition_mode" => string(mala_precondition_mode),
                "mala_precondition_diag" => mala_precondition_diag,
                "mala_metric_ridge" => mala_metric_ridge,
                "mala_metric_max_cholesky_tries" => mala_metric_max_cholesky_tries,
                "mala_stepsize_schedule" => string(mala_stepsize_schedule),
                "mala_sgld_c0" => mala_sgld_c0,
                "mala_sgld_target_accept" => mala_sgld_target_accept,
                "mala_sgld_accept_low" => mala_sgld_accept_low,
                "mala_sgld_accept_high" => mala_sgld_accept_high,
                "mala_sgld_adapt_c0" => mala_sgld_adapt_c0,
                "mala_sgld_adapt_every" => mala_sgld_adapt_every,
                "mala_sgld_c0_min" => mala_sgld_c0_min,
                "mala_sgld_c0_max" => mala_sgld_c0_max,
                "prox_block_step_size" => prox_block_step_size,
                "prox_block_min_step_size" => prox_block_min_step_size,
                "prox_block_max_step_size" => prox_block_max_step_size,
                "prox_block_adapt_every" => prox_block_adapt_every,
                "prox_block_divergence_threshold" => prox_block_divergence_threshold,
                "prox_block_metric" => string(prox_block_metric),
                "use_car" => actual_use_car,
                "car_distance" => string(car_distance),
                "car_weight_quantile" => car_weight_quantile,
                "car_precision" => car_precision,
                "sample_car_strength" => actual_sample_car_strength,
                "car_strength_prior_a" => car_strength_prior_a,
                "car_strength_prior_b" => car_strength_prior_b,
                "car_degree_scale" => string(car_degree_scale),
                "car_kappa_init" => car_kappa_init,
                "input" => input,
                "comp_cols" => join(comp_cols_arg, ","),
                "component_cols" => join(component_cols, ","),
                "exclude_cols" => join(exclude_cols, ","),
                "dropped_survival_rows" => dropped_survival_rows,
            ), prior_metadata),
            settings = merge(Dict(
                "n_samples" => n_samples,
                "warmup" => warmup,
                "thin" => thin,
                "chains" => chains,
                "multichains" => multichains,
                "target_accept" => target_accept,
                "max_depth" => max_depth,
                "hmc_n_leapfrog" => hmc_n_leapfrog,
                "hmc_step_size" => hmc_step_size,
                "seed" => seed,
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
                "mala_precondition" => mala_precondition,
                "mala_precondition_mode" => string(mala_precondition_mode),
                "mala_precondition_diag" => mala_precondition_diag,
                "mala_metric_ridge" => mala_metric_ridge,
                "mala_metric_max_cholesky_tries" => mala_metric_max_cholesky_tries,
                "mala_stepsize_schedule" => string(mala_stepsize_schedule),
                "mala_sgld_c0" => mala_sgld_c0,
                "mala_sgld_target_accept" => mala_sgld_target_accept,
                "mala_sgld_accept_low" => mala_sgld_accept_low,
                "mala_sgld_accept_high" => mala_sgld_accept_high,
                "mala_sgld_adapt_c0" => mala_sgld_adapt_c0,
                "mala_sgld_adapt_every" => mala_sgld_adapt_every,
                "mala_sgld_c0_min" => mala_sgld_c0_min,
                "mala_sgld_c0_max" => mala_sgld_c0_max,
                "prox_block_step_size" => prox_block_step_size,
                "prox_block_min_step_size" => prox_block_min_step_size,
                "prox_block_max_step_size" => prox_block_max_step_size,
                "prox_block_adapt_every" => prox_block_adapt_every,
                "prox_block_divergence_threshold" => prox_block_divergence_threshold,
                "prox_block_metric" => string(prox_block_metric),
                "use_car" => actual_use_car,
                "car_distance" => string(car_distance),
                "car_weight_quantile" => car_weight_quantile,
                "car_precision" => car_precision,
                "sample_car_strength" => actual_sample_car_strength,
                "car_strength_prior_a" => car_strength_prior_a,
                "car_strength_prior_b" => car_strength_prior_b,
                "car_degree_scale" => string(car_degree_scale),
                "car_kappa_init" => car_kappa_init,
                "pseudocount" => pseudocount,
                "standardize" => standardize,
                "tie_method" => string(tie_method),
                "p0_alpha" => p0_alpha,
                "comp_cols" => join(comp_cols_arg, ","),
                "component_cols" => join(component_cols, ","),
                "exclude_cols" => join(exclude_cols, ","),
                "dropped_survival_rows" => dropped_survival_rows,
            ), prior_metadata),
            posterior_summary = summary,
        )
        if diagnose
            diagnose_run(
                run_dir;
                output_dir = joinpath(run_dir, "diagnose"),
                top_k_beta = diagnose_top_k_beta,
                max_lag = diagnose_max_lag,
                formats = diagnose_formats,
            )
        end
        coefficient_plot(
            summary,
            joinpath(figs_dir, "coef_$prior.pdf");
            title = "$(basename(input)) coefficient estimates: $prior",
            estimate = :mean,
        )
        coefficient_plot(
            summary,
            joinpath(figs_dir, "coef_median_$prior.pdf");
            title = "$(basename(input)) median coefficient estimates: $prior",
            estimate = :median,
        )
        coefficient_plot(
            summary,
            joinpath(figs_dir, "coef_map_$prior.pdf");
            title = "$(basename(input)) MAP coefficient estimates: $prior",
            estimate = :map,
        )

        metrics = copy(fit.metrics)
        metrics.prior .= prior
        metrics.p0 .= fit.p0
        add_prior_hyperparameter_columns!(metrics, prior_kwargs)
        append!(all_metrics, metrics; cols = :union)
    end

    add_elpd_diffs!(all_metrics; group_cols = [:estimate])
    write_table(joinpath(tables_dir, "model_metrics.csv"), all_metrics)
    write_table(joinpath(tables_dir, "component_columns.csv"), DataFrame(component = string.(component_cols)))
    metrics_plot(all_metrics, joinpath(figs_dir, "model_metrics.pdf"); title = "T1D model metrics")
    println("Real-data outputs written to $(abspath(outdir))")
end

main()
