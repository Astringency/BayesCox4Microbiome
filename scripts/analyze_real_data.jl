#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "analysis_common.jl"))

function detect_taxa_columns(df::DataFrame, time_col::Symbol, status_col::Symbol, exclude_cols)
    excluded = Set(vcat([time_col, status_col], Symbol.(exclude_cols)))
    return [c for c in names(df) if Symbol(c) ∉ excluded && eltype(df[!, c]) <: Union{Missing,Real}]
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
    priors = split_strings(arg(args, "priors", "hs,cox-rhs,lasso,sslasso"))
    n_samples = arg(args, "samples", 500)
    warmup = arg(args, "warmup", 200)
    seed = arg(args, "seed", 1)
    use_car = arg(args, "use-car", true)
    car_distance = Symbol(arg(args, "car-distance", "bray"))

    tables_dir = ensure_dir(joinpath(outdir, "tables"))
    figs_dir = ensure_dir(joinpath(outdir, "figures"))

    df = CSV.read(input, DataFrame)
    taxa_cols = detect_taxa_columns(df, time_col, status_col, exclude_cols)
    isempty(taxa_cols) && throw(ArgumentError("no taxa abundance columns detected."))
    X = matrix_without_missing(df, taxa_cols)
    time = Float64.(df[!, time_col])
    event = Int.(df[!, status_col])
    p0 = max(1, round(Int, min(0.1 * length(taxa_cols), count(!=(0), event) / 2)))

    all_metrics = DataFrame()
    for prior in priors
        fit = fit_and_evaluate(
            X,
            time,
            event;
            prior_name = prior,
            n_samples = n_samples,
            warmup = warmup,
            seed = seed,
            use_car = use_car && occursin("hs", lowercase(prior)),
            car_distance = car_distance,
            p0 = p0,
            realdata = true,
        )
        summary = copy(fit.summary)
        summary.variable = string.(taxa_cols)
        summary.selected = (summary.lower .* summary.upper) .> 0
        write_table(joinpath(tables_dir, "coef_$prior.csv"), summary)
        coefficient_plot(
            summary,
            joinpath(figs_dir, "coef_$prior.png");
            title = "T1D coefficient estimates: $prior",
        )

        metrics = copy(fit.metrics)
        metrics.prior .= prior
        append!(all_metrics, metrics; cols = :union)
    end

    write_table(joinpath(tables_dir, "model_metrics.csv"), all_metrics)
    write_table(joinpath(tables_dir, "taxa_columns.csv"), DataFrame(taxon = string.(taxa_cols)))
    metrics_plot(all_metrics, joinpath(figs_dir, "model_metrics.png"); title = "T1D model metrics")
    println("Real-data outputs written to $(abspath(outdir))")
end

main()
