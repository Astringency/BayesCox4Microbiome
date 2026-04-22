#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

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
    target_accept = arg(args, "target-accept", 0.8)
    seed = arg(args, "seed", 1)
    use_car = arg(args, "use-car", true)
    car_distance = Symbol(arg(args, "car-distance", "bray"))
    p0_arg = parse_optional_p0(args; default = nothing)
    p0_arg === :truth && throw(ArgumentError("--p0 truth is only valid for simulation."))
    p0_alpha = arg(args, "p0-alpha", 0.05)

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
            target_accept = target_accept,
            seed = seed,
            use_car = use_car && occursin("hs", lowercase(prior)),
            car_distance = car_distance,
            p0 = p0_arg,
            p0_alpha = p0_alpha,
            realdata = true,
        )
        summary = copy(fit.summary)
        summary.variable = string.(taxa_cols)
        summary.selected = (summary.lower .* summary.upper) .> 0
        write_table(joinpath(tables_dir, "coef_$prior.csv"), summary)
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

    write_table(joinpath(tables_dir, "model_metrics.csv"), all_metrics)
    write_table(joinpath(tables_dir, "taxa_columns.csv"), DataFrame(taxon = string.(taxa_cols)))
    metrics_plot(all_metrics, joinpath(figs_dir, "model_metrics.pdf"); title = "T1D model metrics")
    println("Real-data outputs written to $(abspath(outdir))")
end

main()
