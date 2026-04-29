#!/usr/bin/env julia

using Pkg

function _activate_repo_project()
    project = abspath(joinpath(@__DIR__, "..", "Project.toml"))
    active = Base.active_project()
    active !== project && Pkg.activate(dirname(project))
end

_activate_repo_project()

using BayesCox4Micro

function parse_cli_args(args = ARGS)
    out = Dict{String,String}()
    positional = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--")
            key = a[3:end]
            if occursin("=", key)
                k, v = split(key, "="; limit = 2)
                out[k] = v
            elseif i < length(args) && !startswith(args[i + 1], "--")
                out[key] = args[i + 1]
                i += 1
            else
                out[key] = "true"
            end
        else
            push!(positional, a)
        end
        i += 1
    end
    !isempty(positional) && !haskey(out, "input") && (out["input"] = first(positional))
    return out
end

function main()
    args = parse_cli_args()
    input = get(args, "input", joinpath("results", "runs", "example"))
    output = get(args, "output", joinpath(input, "diagnose"))
    params = haskey(args, "params") ? args["params"] : nothing
    top_k_beta = parse(Int, get(args, "top-k-beta", "10"))
    max_lag = parse(Int, get(args, "max-lag", "50"))
    rhat_threshold = parse(Float64, get(args, "rhat-threshold", "1.01"))
    formats = get(args, "format", "png,pdf")

    result = diagnose_run(
        input;
        output_dir = output,
        params = params,
        top_k_beta = top_k_beta,
        max_lag = max_lag,
        rhat_threshold = rhat_threshold,
        formats = formats,
    )
    println("Diagnostics written to $(abspath(result.output_dir))")
end

main()
