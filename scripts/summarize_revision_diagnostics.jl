#!/usr/bin/env julia
# Read-only historical reevaluation. This script contains no sampling calls.
using BayesCox4Micro, CSV, DataFrames, JSON, Statistics

function revision_options(args)
    opts = Dict{String,String}()
    iseven(length(args)) || error("use --name value pairs")
    for i in 1:2:length(args)
        startswith(args[i], "--") || error("invalid option $(args[i])")
        opts[args[i][3:end]] = args[i+1]
    end
    allowed = Set(["roots", "outdir", "priors", "max-runs", "resume"])
    all(k -> k in allowed, keys(opts)) || error("unknown option")
    haskey(opts, "roots") && haskey(opts, "outdir") || error("--roots and --outdir are required")
    return opts
end

function revision_summary_main(args = ARGS)
    B = BayesCox4Micro
    opts = revision_options(args)
    roots = abspath.(split(opts["roots"], ','))
    outdir = abspath(opts["outdir"])
    outdir in roots && error("reevaluation output must differ from historical input roots")
    mkpath(outdir)
    priors = Set(split(get(opts, "priors", "cox-rhs,prox-sslasso"), ','))
    maxruns = parse(Int, get(opts, "max-runs", string(typemax(Int))))
    resume = lowercase(get(opts, "resume", "true")) == "true"
    inventories, diagnostic_tables = DataFrame[], DataFrame[]
    counter = 0
    for (rootid, root) in enumerate(roots)
        isdir(root) || error("input root does not exist: $root")
        for (dir, dirs, files) in walkdir(root)
            filter!(d -> !startswith(d, ".") && !(d in ("plots", "figures", "diagnose", "tables", "test_data")) &&
                abspath(joinpath(dir, d)) != outdir, dirs)
            "metadata.json" in files || continue
            counter >= maxruns && break
            meta = JSON.parsefile(joinpath(dir, "metadata.json"))
            method = string(get(meta, "method_name", get(meta, "prior_name", get(meta, "prior", "unknown"))))
            method in priors || continue
            "samples_wide.csv" in files || continue
            counter += 1
            samples_path = joinpath(dir, "samples_wide.csv")
            dest = joinpath(outdir, "root" * lpad(rootid, 2, '0'), relpath(dir, root))
            mkpath(dest)
            fingerprint = Dict("schema" => 2, "source" => samples_path,
                "samples_bytes" => filesize(samples_path), "samples_mtime" => mtime(samples_path),
                "metadata_mtime" => mtime(joinpath(dir, "metadata.json")))
            manifestpath = joinpath(dest, "reevaluation_manifest.json")
            complete = resume && isfile(manifestpath) && JSON.parsefile(manifestpath) == fingerprint &&
                isfile(joinpath(dest, "diagnostics_summary.csv"))
            status, errmsg = "complete", ""
            transition_path = joinpath(dir, "transition_diagnostics.csv")
            diag_path = joinpath(dir, "diagnostics_summary.csv")
            costs = isfile(diag_path) ? read_diagnostics(dir; category = "runtime") : DataFrame()
            diag = isempty(costs) ? Dict() : Dict(String(k) => v for (k,v) in pairs(first(eachrow(costs))) if !ismissing(v))
            runtime = get(diag, "total_seconds", get(meta, "runtime", missing))
            runtime === nothing && (runtime = missing)
            id = (source = dir, method = method, constraint = get(meta, "constraint", missing),
                prox_sum0_lambda = get(meta, "prox_sum0_lambda", missing), n = get(meta, "n", missing),
                p = get(meta, "p", missing), nonzero = get(meta, "nonzero_count", missing),
                baseline = get(meta, "baseline", missing), replicate = get(meta, "replicate_id", missing),
                p0 = get(meta, "p0", missing), saved_sample_thin = get(meta, "saved_sample_thin", missing),
                total_seconds = runtime)
            try
                if !complete
                    wide = CSV.read(samples_path, DataFrame)
                    transitions = isfile(transition_path) ? CSV.read(transition_path, DataFrame) : DataFrame()
                    B._save_numerical_diagnostics(wide, dest; transitions, runtime = diag)
                    B._write_json(manifestpath, fingerprint)
                end
                df = read_diagnostics(dest)
                for (k,v) in pairs(id)
                    df[!, k] = fill(v === nothing ? missing : v, nrow(df))
                end
                push!(diagnostic_tables, df)
            catch err
                status, errmsg = "failed", sprint(showerror, err)
                @warn "Historical reevaluation failed" dir exception=(err, catch_backtrace())
            end
            inventory_row = merge(id, (status = status, error = errmsg, output_dir = dest,
                transition_diagnostics_available = isfile(transition_path),
                transition_missing_reason = isfile(transition_path) ? "" : "not saved; cannot reconstruct from posterior draws"))
            push!(inventories, DataFrame([inventory_row]))
            CSV.write(joinpath(outdir, "run_inventory.csv"), vcat(inventories...; cols = :union))
            println("[$counter] $status $dir")
            flush(stdout)
        end
    end
    !isempty(diagnostic_tables) && CSV.write(joinpath(outdir, "diagnostics_all.csv"), vcat(diagnostic_tables...; cols = :union))
    isempty(inventories) && error("no matching saved runs found")
    any(df -> any(==("failed"), df.status), inventories) && error("some reevaluations failed; inspect run_inventory.csv")
    println("Reevaluated $counter saved runs. No model was resampled.")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    revision_summary_main()
end
