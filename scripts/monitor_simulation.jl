#!/usr/bin/env julia

using Dates
using Printf

const DEFAULT_ROOT = joinpath("results", "simulation", "softcenter")
const DEFAULT_PRIORS = [
    "hs",
    "rhs",
    "new-rhs",
    "est-rhs",
    "cox-hs",
    "cox-rhs",
    "lasso",
    "sslasso",
    "prox-lasso",
    "prox-sslasso"
]

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

function split_csv(value::AbstractString)
    return [strip(x) for x in split(value, ",") if !isempty(strip(x))]
end

function setting_from_dirname(name::AbstractString)
    m = match(r"^n(\d+)_p(\d+)_nonzero(\d+)_(.+)$", name)
    m === nothing && return nothing
    return (
        n = parse(Int, m.captures[1]),
        p = parse(Int, m.captures[2]),
        nonzero = parse(Int, m.captures[3]),
        baseline = m.captures[4],
    )
end

function run_from_dirname(name::AbstractString, setting)
    prefix = "rep"
    startswith(name, prefix) || return nothing
    suffix = "_$(setting.baseline)_n$(setting.n)_p$(setting.p)_nz$(setting.nonzero)"
    endswith(name, suffix) || return nothing
    middle = name[(lastindex(prefix) + 1):(lastindex(name) - lastindex(suffix))]
    m = match(r"^(\d+)_(.+)$", middle)
    m === nothing && return nothing
    return (rep = parse(Int, m.captures[1]), prior = m.captures[2])
end

function file_complete(path::AbstractString)
    return isfile(path) && filesize(path) > 0
end

function sampling_complete(run_dir::AbstractString)
    return file_complete(joinpath(run_dir, "samples_wide.csv")) ||
           file_complete(joinpath(run_dir, "result_bundle.jld2"))
end

function diagnosis_complete(run_dir::AbstractString)
    return file_complete(joinpath(run_dir, "diagnose", "diagnostics_summary.csv"))
end

function metrics_complete(run_dir::AbstractString)
    return file_complete(joinpath(run_dir, "posterior_summary.csv")) &&
           file_complete(joinpath(run_dir, "metrics.csv")) &&
           file_complete(joinpath(run_dir, "false_discovery.csv"))
end

complete_run(run_dir::AbstractString) = sampling_complete(run_dir) && metrics_complete(run_dir)

function collect_status(root::AbstractString, priors::Vector{String})
    rows = NamedTuple[]
    found_priors = Set(priors)
    isdir(root) || return rows, sort!(collect(found_priors))

    for setting_entry in sort(readdir(root; join = true))
        isdir(setting_entry) || continue
        setting = setting_from_dirname(basename(setting_entry))
        setting === nothing && continue

        counts = Dict(prior => Set{Int}() for prior in priors)
        runs_dir = joinpath(setting_entry, "runs")
        if isdir(runs_dir)
            for run_entry in readdir(runs_dir; join = true)
                isdir(run_entry) || continue
                run = run_from_dirname(basename(run_entry), setting)
                run === nothing && continue
                push!(found_priors, run.prior)
                !haskey(counts, run.prior) && (counts[run.prior] = Set{Int}())
                complete_run(run_entry) && push!(counts[run.prior], run.rep)
            end
        end

        push!(rows, (
            n = setting.n,
            p = setting.p,
            nonzero = setting.nonzero,
            baseline = setting.baseline,
            counts = counts,
        ))
    end

    all_priors = sort!(collect(found_priors))
    return rows, all_priors
end

function table_lines(rows, priors::Vector{String}; root::AbstractString)
    lines = String[]
    push!(lines, "Simulation progress monitor")
    push!(lines, "Generated: $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))")
    push!(lines, "Root: $(abspath(root))")
    push!(lines, "")

    if isempty(rows)
        push!(lines, "No setting directories found.")
        return lines
    end

    headers = vcat(["n", "p", "nonzero", "baseline"], priors)
    data = Vector{Vector{String}}()
    for row in sort(rows; by = r -> (r.n, r.p, r.nonzero, r.baseline))
        counts = [string(length(get(row.counts, prior, Set{Int}()))) for prior in priors]
        push!(data, vcat(string.([row.n, row.p, row.nonzero, row.baseline]), counts))
    end

    widths = [length(h) for h in headers]
    for r in data
        for i in eachindex(r)
            widths[i] = max(widths[i], length(r[i]))
        end
    end

    function fmt_row(values)
        parts = String[]
        for i in eachindex(values)
            push!(parts, lpad(values[i], widths[i]))
        end
        return join(parts, "  ")
    end

    push!(lines, fmt_row(headers))
    push!(lines, join(["-" ^ w for w in widths], "  "))
    append!(lines, fmt_row.(data))
    push!(lines, "")
    push!(lines, "A repeat is counted only when sampling output, posterior_summary.csv, metrics.csv, and false_discovery.csv exist.")
    return lines
end

function main()
    args = parse_cli_args()
    root = get(args, "root", DEFAULT_ROOT)
    priors = haskey(args, "priors") ? split_csv(args["priors"]) : copy(DEFAULT_PRIORS)

    rows, all_priors = collect_status(root, priors)
    lines = table_lines(rows, all_priors; root = root)
    output = join(lines, "\n")

    println(output)
    mkpath(root)
    write(joinpath(root, "running.txt"), output * "\n")
end

main()
