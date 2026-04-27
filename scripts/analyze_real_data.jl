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

function detect_taxa_columns(df::DataFrame, time_col::Symbol, status_col::Symbol, exclude_cols)
    excluded = Set(vcat([time_col, status_col], Symbol.(exclude_cols)))
    return [c for c in names(df) if !(Symbol(c) in excluded) && eltype(df[!, c]) <: Union{Missing,Real}]
end

function matrix_without_missing(df::DataFrame, cols)
    mat = Matrix{Float64}(coalesce.(Matrix(df[:, cols]), 0.0))
    all(>=(0), mat) || throw(ArgumentError("taxa abundance columns must be nonnegative."))
    return mat
end

function main()
    args = parse_cli_args()
    input = arg(args, "input", joinpath("data", "T1D.csv"))
    outdir = arg(args, "outdir", joinpath("results", "realdata", "T1D"))
    time_col = Symbol(arg(args, "time-col", "time"))
    status_col = Symbol(arg(args, "status-col", "status"))
    exclude_cols = split_strings(arg(args, "exclude-cols", "Sex,Antibiotics"))
    priors = split_strings(arg(args, "priors", "hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso"))
    sampler = Symbol(arg(args, "sampler", "nuts"))
    n_samples = arg(args, "samples", 500)
    warmup = arg(args, "warmup", 200)
    chains = arg(args, "chains", 4)
    multichains = arg(args, "multichains", false)
    target_accept = arg(args, "target-accept", 0.8)
    seed = arg(args, "seed", 1)
    use_car = arg(args, "use-car", true)
    progress = arg(args, "progress", true)
    car_distance = Symbol(arg(args, "car-distance", "bray"))
    p0_arg = parse_optional_p0(args; default = nothing)
    p0_arg === :truth && throw(ArgumentError("--p0 truth is only valid for simulation."))
    p0_alpha = arg(args, "p0-alpha", 0.05)
    constraint = Symbol(arg(args, "constraint", "soft_center"))
    soft_center_scale = haskey(args, "soft-center-scale") ? parse(Float64, args["soft-center-scale"]) : nothing
    prox_sum0_lambda = arg(args, "prox-sum0-lambda", 1e-3)
    prox_l1_lambda = arg(args, "prox-l1-lambda", 1e-3)
    prox_root_tol = arg(args, "prox-root-tol", 1e-10)
    prox_root_maxiter = arg(args, "prox-root-maxiter", 100)
    mala_step_size = arg(args, "mala-step-size", 1e-4)
    mala_target_accept = arg(args, "mala-target-accept", 0.574)
    mala_adapt_every = arg(args, "mala-adapt-every", 50)
    mala_min_step = arg(args, "mala-min-step", 1e-8)
    mala_max_step = arg(args, "mala-max-step", 1.0)

    tables_dir = ensure_dir(joinpath(outdir, "tables"))
    figs_dir = ensure_dir(joinpath(outdir, "figures"))

    df = CSV.read(input, DataFrame)
    taxa_cols = detect_taxa_columns(df, time_col, status_col, exclude_cols)
    isempty(taxa_cols) && throw(ArgumentError("no taxa abundance columns detected."))
    X = matrix_without_missing(df, taxa_cols)
    time = Float64.(df[!, time_col])
    event = Int.(df[!, status_col])
    if p0_arg === nothing
        screen_table, screen = univariate_screen_table(X, time, event; names = taxa_cols, alpha = p0_alpha)
        screen_table.p0 .= screen.p0
        screen_table.alpha .= screen.alpha
        write_table(joinpath(tables_dir, "univariate_cox_screen.csv"), screen_table)
    end

    all_metrics = DataFrame()
    for prior in priors
        fit = fit_and_evaluate(
            X,
            time,
            event;
            prior_name = prior,
            sampler = sampler,
            n_samples = n_samples,
            warmup = warmup,
            chains = chains,
            multichains = multichains,
            target_accept = target_accept,
            seed = seed,
            use_car = use_car && occursin("hs", lowercase(prior)),
            car_distance = car_distance,
            p0 = p0_arg,
            p0_alpha = p0_alpha,
            constraint = constraint,
            soft_center_scale = soft_center_scale,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
            mala_step_size = mala_step_size,
            mala_target_accept = mala_target_accept,
            mala_adapt_every = mala_adapt_every,
            mala_min_step = mala_min_step,
            mala_max_step = mala_max_step,
            realdata = true,
            progress = progress,
        )
        summary = copy(fit.summary)
        summary.variable = string.(taxa_cols)
        summary.selected = (summary.lower .* summary.upper) .> 0
        write_table(joinpath(tables_dir, "coef_$prior.csv"), summary)
        run_id = "realdata_$(prior)"
        save_run_outputs(
            fit.result;
            output_dir = joinpath(outdir, "runs", run_id),
            run_id = run_id,
            metadata = Dict(
                "dataset_name" => basename(input),
                "method_name" => prior,
                "prior_name" => prior,
                "sampler" => string(sampler),
                "p0" => fit.p0,
                "input" => input,
            ),
            settings = Dict(
                "n_samples" => n_samples,
                "warmup" => warmup,
                "chains" => chains,
                "multichains" => multichains,
                "target_accept" => target_accept,
                "seed" => seed,
                "thin" => 1,
                "constraint" => string(constraint),
                "soft_center_scale" => soft_center_scale,
                "prox_sum0_lambda" => prox_sum0_lambda,
                "prox_l1_lambda" => prox_l1_lambda,
                "prox_root_tol" => prox_root_tol,
                "prox_root_maxiter" => prox_root_maxiter,
                "mala_step_size" => mala_step_size,
                "mala_target_accept" => mala_target_accept,
                "mala_adapt_every" => mala_adapt_every,
                "mala_min_step" => mala_min_step,
                "mala_max_step" => mala_max_step,
                "use_car" => use_car && occursin("hs", lowercase(prior)),
                "car_distance" => string(car_distance),
                "p0_alpha" => p0_alpha,
            ),
            posterior_summary = summary,
        )
        coefficient_plot(
            summary,
            joinpath(figs_dir, "coef_$prior.pdf");
            title = "T1D coefficient estimates: $prior",
            estimate = :mean,
        )
        coefficient_plot(
            summary,
            joinpath(figs_dir, "coef_median_$prior.pdf");
            title = "T1D median coefficient estimates: $prior",
            estimate = :median,
        )
        coefficient_plot(
            summary,
            joinpath(figs_dir, "coef_map_$prior.pdf");
            title = "T1D MAP coefficient estimates: $prior",
            estimate = :map,
        )

        metrics = copy(fit.metrics)
        metrics.prior .= prior
        metrics.p0 .= fit.p0
        append!(all_metrics, metrics; cols = :union)
    end

    add_elpd_diffs!(all_metrics; group_cols = [:estimate])
    write_table(joinpath(tables_dir, "model_metrics.csv"), all_metrics)
    write_table(joinpath(tables_dir, "taxa_columns.csv"), DataFrame(taxon = string.(taxa_cols)))
    metrics_plot(all_metrics, joinpath(figs_dir, "model_metrics.pdf"); title = "T1D model metrics")
    println("Real-data outputs written to $(abspath(outdir))")
end

main()
