#!/usr/bin/env julia

using CSV
using DataFrames
using Dates

function parse_cli_args(args = ARGS)
    out = Dict{String,String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key = arg[3:end]
            if occursin("=", key)
                k, v = split(key, "="; limit = 2)
                out[k] = v
            elseif i < length(args) && !startswith(args[i + 1], "--")
                out[key] = args[i + 1]
                i += 1
            else
                out[key] = "true"
            end
        end
        i += 1
    end
    return out
end

function ensure_column!(df::DataFrame, col::Symbol)
    col in propertynames(df) || (df[!, col] = fill(missing, nrow(df)))
    return df
end

function select_existing(df::DataFrame, cols)
    for col in cols
        ensure_column!(df, col)
    end
    return df[:, cols]
end

function sort_existing!(df::DataFrame, cols)
    keys = [col for col in cols if col in propertynames(df)]
    isempty(keys) || sort!(df, keys)
    return df
end

function read_summary(path::AbstractString)
    isfile(path) || error("Missing $path. Run scripts/summarize_simulation_results.jl first.")
    return CSV.read(path, DataFrame)
end

function prefixed_subset(df::DataFrame, keys, cols, prefix::AbstractString)
    keep = Symbol[]
    for col in keys
        col in propertynames(df) && push!(keep, col)
    end
    for col in cols
        col in propertynames(df) && push!(keep, col)
    end
    out = df[:, unique(keep)]
    rename_map = Dict{Symbol,Symbol}()
    keyset = Set(keys)
    for col in propertynames(out)
        col in keyset && continue
        rename_map[col] = Symbol(prefix, col)
    end
    rename!(out, rename_map)
    return out
end

function main()
    args = parse_cli_args()
    root = get(args, "root", joinpath("results", "simulation"))

    model = read_summary(joinpath(root, "model_metrics_summary.csv"))
    sampling = read_summary(joinpath(root, "sampling_diagnostics_summary.csv"))
    parameter = read_summary(joinpath(root, "parameter_mcmc_diagnostics_summary.csv"))

    join_keys = [:root, :n, :p, :nonzero_count, :baseline, :prior, :sampler]
    sampling_cols = [
        :run_count,
        :complete_run_count,
        :acceptance_rate_mean,
        :total_seconds_mean,
    ]
    beta_cols = [
        :parameter_row_count,
        :rhat_mean,
        :rhat_median,
        :rhat_max,
        :rhat_over_threshold_count,
        :rhat_threshold,
        :ess_bulk_min,
        :ess_bulk_median,
    ]

    sampling_part = prefixed_subset(sampling, join_keys, sampling_cols, "sampling_")
    beta = :parameter_group in propertynames(parameter) ?
        parameter[string.(parameter.parameter_group) .== "beta", :] :
        DataFrame()
    beta_part = prefixed_subset(beta, join_keys, beta_cols, "beta_mcmc_")

    rows = leftjoin(model, sampling_part; on = [c for c in join_keys if c in propertynames(model) && c in propertynames(sampling_part)], matchmissing = :equal)
    rows = leftjoin(rows, beta_part; on = [c for c in join_keys if c in propertynames(rows) && c in propertynames(beta_part)], matchmissing = :equal)

    selected_cols = [
        :n,
        :p,
        :nonzero_count,
        :baseline,
        :sampler,
        :prior,
        :estimate,
        :run_count,
        :sampling_complete_run_count,
        :MSE_mean,
        :MSE_sd,
        :FN_mean,
        :FN_sd,
        :FP_mean,
        :FP_sd,
        :FDR_mean,
        :FDR_sd,
        :beta_mcmc_rhat_mean,
        :beta_mcmc_rhat_median,
        :beta_mcmc_rhat_max,
        :beta_mcmc_rhat_over_threshold_count,
        :beta_mcmc_rhat_threshold,
        :sampling_acceptance_rate_mean,
        :sampling_total_seconds_mean,
    ]
    out = select_existing(rows, selected_cols)
    sort_existing!(out, [:sampler, :nonzero_count, :baseline, :n, :estimate])

    all_path = joinpath(root, "requested_mse_fn_fp_rhat_all_estimates.csv")
    CSV.write(all_path, out)

    mean_rows = out[string.(out.estimate) .== "mean", :]
    mean_path = joinpath(root, "requested_mse_fn_fp_rhat_mean_estimate.csv")
    CSV.write(mean_path, mean_rows)

    report = [
        "Requested p=150 simulation summary",
        "Generated: $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))",
        "Root: $(abspath(root))",
        "Rows in all-estimates table: $(nrow(out))",
        "Rows in mean-estimate table: $(nrow(mean_rows))",
        "",
        "Output files:",
        "- $all_path",
        "- $mean_path",
    ]
    report_path = joinpath(root, "requested_mse_fn_fp_rhat_report.txt")
    write(report_path, join(report, "\n") * "\n")
    println(join(report, "\n"))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
