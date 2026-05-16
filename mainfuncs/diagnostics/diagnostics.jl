using CSV
using DataFrames
using Dates

const _PKG_UUIDS = Dict(
    :JLD2 => "033835bb-8acc-5ee8-8aae-3f567f8a3819",
    :JSON => "682c06a0-de6a-54ab-a142-c8b1cf79cde6",
    :MCMCChains => "c7f686f2-ff18-58e9-bc7b-31028e88f75d",
    :Plots => "91a5bcdd-55d7-5caf-9e0b-520d859cae80",
    :StatsPlots => "f3b207a7-027a-5e70-b257-86293d7955fd",
)

function _require_pkg(name::Symbol)
    uuid = _PKG_UUIDS[name]
    return Base.require(Base.PkgId(Base.UUID(uuid), String(name)))
end

function _configure_headless_gr!()
    if !haskey(ENV, "GKSwstype") && !haskey(ENV, "DISPLAY")
        ENV["GKSwstype"] = "100"
    end
    return nothing
end

_call_latest(f, args...; kwargs...) = Base.invokelatest(f, args...; kwargs...)

function _is_chains_object(x)
    MCMCChains = _require_pkg(:MCMCChains)
    return x isa getproperty(MCMCChains, :Chains)
end

function _safe_string(x)
    x === missing && return missing
    x === nothing && return nothing
    x isa Symbol && return String(x)
    x isa AbstractString && return String(x)
    x isa Type && return string(x)
    return x
end

function _jsonable(x)
    x === missing && return nothing
    x === nothing && return nothing
    x isa Symbol && return String(x)
    x isa AbstractString && return String(x)
    x isa Number && return isfinite(Float64(x)) ? x : string(x)
    x isa Bool && return x
    x isa Type && return string(x)
    x isa AbstractVector && return [_jsonable(v) for v in x]
    x isa Tuple && return [_jsonable(v) for v in x]
    if x isa NamedTuple
        return Dict(String(k) => _jsonable(v) for (k, v) in pairs(x))
    elseif x isa AbstractDict
        return Dict(String(k) => _jsonable(v) for (k, v) in pairs(x))
    elseif x isa SamplingDiagnostics
        return _jsonable(sampling_diagnostics_dict(x))
    end
    return string(x)
end

function _write_json(path::AbstractString, x)
    JSON = _require_pkg(:JSON)
    open(path, "w") do io
        _call_latest(getproperty(JSON, :print), io, _jsonable(x), 4)
        write(io, "\n")
    end
    return path
end

function _csv_value(x)
    x === nothing && return missing
    x === missing && return missing
    x isa Symbol && return String(x)
    x isa Type && return string(x)
    (x isa AbstractVector || x isa Tuple || x isa NamedTuple || x isa AbstractDict) && return string(_jsonable(x))
    return x
end

function _write_csv(path::AbstractString, table)
    return CSV.write(path, table; transform = (col, val) -> _csv_value(val))
end

function _git_commit_hash()
    try
        return readchomp(`git rev-parse HEAD`)
    catch
        return missing
    end
end

function _prior_name(prior)
    return string(nameof(typeof(prior)))
end

function sampling_diagnostics_dict(d::SamplingDiagnostics)
    return Dict(
        "warmup_seconds" => d.warmup_seconds,
        "sample_seconds" => d.sample_seconds,
        "total_seconds" => d.total_seconds,
        "allocated_bytes" => d.allocated_bytes,
        "allocated_mb" => d.allocated_bytes / 2.0^20,
        "gc_seconds" => d.gc_seconds,
        "gc_percent" => d.gc_percent,
    )
end

function _sampler_acceptance_metadata(result::CoxSamplerResult)
    if result.samples isa MHSamples
        return Dict(
            "acceptance_rate" => result.samples.acceptance_rate,
            "warmup_acceptance_rate" => result.samples.warmup_acceptance_rate,
            "final_step_size" => result.samples.final_step,
        )
    elseif result.samples isa MultiMHSamples
        samples = result.samples.chains
        return Dict(
            "acceptance_rate" => mean([s.acceptance_rate for s in samples]),
            "warmup_acceptance_rate" => mean([s.warmup_acceptance_rate for s in samples]),
            "acceptance_rate_by_chain" => [s.acceptance_rate for s in samples],
            "warmup_acceptance_rate_by_chain" => [s.warmup_acceptance_rate for s in samples],
            "final_step_size_by_chain" => [s.final_step for s in samples],
        )
    end
    return Dict(
        "acceptance_rate" => missing,
        "warmup_acceptance_rate" => missing,
        "final_step_size" => missing,
    )
end

function _base_metadata(result::CoxSamplerResult; run_id = "run", metadata = Dict(), settings = Dict())
    data = result.data
    git_sha = _git_commit_hash()
    samples_value = get(settings, "n_samples", get(settings, :n_samples, get(settings, "samples", missing)))
    base = Dict{String,Any}(
        "run_id" => run_id,
        "sampler" => String(result.sampler),
        "prior" => _prior_name(result.prior),
        "constraint" => get(settings, "constraint", get(settings, :constraint, missing)),
        "seed" => get(settings, "seed", get(settings, :seed, missing)),
        "n" => length(data.time),
        "p" => size(data.z, 2),
        "number_of_events" => count(!=(0), data.event),
        "samples" => samples_value,
        "n_samples" => samples_value,
        "warmup" => get(settings, "warmup", get(settings, :warmup, missing)),
        "chains" => get(settings, "chains", get(settings, :chains, missing)),
        "thin" => get(settings, "thin", get(settings, :thin, missing)),
        "use_car" => get(settings, "use_car", get(settings, :use_car, missing)),
        "car_distance" => get(settings, "car_distance", get(settings, :car_distance, missing)),
        "tau_scale_mode" => get(settings, "tau_scale_mode", get(settings, :tau_scale_mode, get(ENV, "BAYESCOX_TAU_CAUCHY_SCALE", "tau0_squared"))),
        "pseudocount" => data.pseudocount,
        "runtime" => result.diagnostics.total_seconds,
        "git_commit_hash" => git_sha,
        "git_sha" => git_sha,
        "created_at" => string(now()),
    )
    merge!(base, _sampler_acceptance_metadata(result))
    merge!(base, Dict(String(k) => _safe_string(v) for (k, v) in pairs(metadata)))
    return base
end

"""
    result_to_chains(result::CoxSamplerResult)

Convert sampler output to `MCMCChains.Chains`. Turing/NUTS `Chains` objects are
returned directly. Built-in `MHSamples` results are converted to a one-chain
`Chains` object with `logposterior` appended as a parameter.
"""
function result_to_chains(result::CoxSamplerResult)
    MCMCChains = _require_pkg(:MCMCChains)
    Chains = getproperty(MCMCChains, :Chains)
    if result.samples isa Chains
        return result.samples
    elseif result.samples isa MultiMHSamples
        return _mhsamples_to_chains(result.samples)
    elseif result.samples isa MHSamples
        draws = result.samples.draws
        names = copy(result.samples.names)
        values = hcat(draws, result.samples.logposterior)
        push!(names, "logposterior")
        arr = reshape(values, size(values, 1), size(values, 2), 1)
        return Chains(arr, Symbol.(names))
    end
    throw(ArgumentError("unsupported samples object for MCMCChains conversion: $(typeof(result.samples))"))
end

as_mcmc_chains(result::CoxSamplerResult) = result_to_chains(result)

function _chains_array_and_names(chains)
    arr = try
        Array(getfield(chains, :value))
    catch
        Array(chains)
    end
    if ndims(arr) == 2
        arr = reshape(arr, size(arr, 1), size(arr, 2), 1)
    elseif ndims(arr) != 3
        throw(ArgumentError("Chains values must be a 2- or 3-dimensional array; got $(ndims(arr)) dimensions with size $(size(arr))."))
    end
    return arr, string.(names(chains))
end

function chains_to_wide_dataframe(chains)
    arr, param_names = _chains_array_and_names(chains)
    n_iter, n_param, n_chain = size(arr)
    df = DataFrame(chain = Int[], iter = Int[], draw = Int[])
    for name in param_names
        df[!, Symbol(name)] = Float64[]
    end
    for c in 1:n_chain, i in 1:n_iter
        row = Dict{Symbol,Any}(:chain => c, :iter => i, :draw => (c - 1) * n_iter + i)
        for j in 1:n_param
            row[Symbol(param_names[j])] = arr[i, j, c]
        end
        push!(df, row; cols = :union)
    end
    return df
end

samples_wide_dataframe(result::CoxSamplerResult) = chains_to_wide_dataframe(result_to_chains(result))

function _thin_wide_samples(wide::DataFrame, sample_thin::Integer)
    sample_thin >= 1 || throw(ArgumentError("sample_thin must be at least 1."))
    sample_thin == 1 && return wide
    if :iter in propertynames(wide)
        keep = mod.(Int.(wide.iter) .- 1, sample_thin) .== 0
    else
        keep = mod.(collect(0:(nrow(wide) - 1)), sample_thin) .== 0
    end
    return wide[keep, :]
end

function dataframe_to_chains(wide::DataFrame)
    MCMCChains = _require_pkg(:MCMCChains)
    Chains = getproperty(MCMCChains, :Chains)
    required = (:chain, :iter)
    all(c -> c in propertynames(wide), required) ||
        throw(ArgumentError("wide samples table must contain chain and iter columns."))
    sorted = sort(wide, [:chain, :iter])
    param_cols = [c for c in propertynames(sorted) if !(c in (:chain, :iter, :draw))]
    chains = sort(unique(Int.(sorted.chain)))
    n_chain = length(chains)
    n_iter = length(unique(sorted.iter))
    arr = Array{Float64}(undef, n_iter, length(param_cols), n_chain)
    for (ci, c) in enumerate(chains)
        rows = sorted[sorted.chain .== c, :]
        nrow(rows) == n_iter ||
            throw(ArgumentError("each chain must have the same number of iterations."))
        for (j, col) in enumerate(param_cols)
            arr[:, j, ci] .= Float64.(rows[!, col])
        end
    end
    return Chains(arr, param_cols)
end

function _posterior_summary_dataframe(chains)
    MCMCChains = _require_pkg(:MCMCChains)
    summary = DataFrame(_call_latest(getproperty(MCMCChains, :summarystats), chains))
    qs = DataFrame(_call_latest(getproperty(MCMCChains, :quantile), chains; q = [0.025, 0.5, 0.975]))
    if :parameters in propertynames(summary) && :parameters in propertynames(qs)
        rename_map = Dict{Symbol,Symbol}()
        for c in propertynames(qs)
            sc = String(c)
            if sc in ("2.5%", "2.5%")
                rename_map[c] = :q025
            elseif sc in ("50.0%", "50%", "median")
                rename_map[c] = :median
            elseif sc in ("97.5%", "97.5%")
                rename_map[c] = :q975
            end
        end
        !isempty(rename_map) && rename!(qs, rename_map)
        keep = [:parameters]
        append!(keep, [c for c in (:q025, :median, :q975) if c in propertynames(qs)])
        return leftjoin(summary, qs[:, keep], on = :parameters)
    end
    return summary
end

function _diagnostics_summary_dataframe(chains; metadata = Dict(), rhat_threshold::Real = 1.01)
    df = _posterior_summary_dataframe(chains)
    :std in propertynames(df) && rename!(df, :std => :sd)
    if !(:parameters in propertynames(df)) && (:parameter in propertynames(df))
        rename!(df, :parameter => :parameters)
    end
    if !(:rhat in propertynames(df))
        df.rhat = fill(missing, nrow(df))
    end
    if !(:ess_bulk in propertynames(df)) && (:ess in propertynames(df))
        rename!(df, :ess => :ess_bulk)
    end
    for (k, v) in pairs(metadata)
        df[!, Symbol(k)] = fill(_csv_value(v), nrow(df))
    end
    df.rhat_threshold = fill(Float64(rhat_threshold), nrow(df))
    return df
end

function save_run_outputs(
    result::CoxSamplerResult;
    output_dir::AbstractString,
    run_id::AbstractString = basename(normpath(output_dir)),
    save_csv::Bool = true,
    save_wide_csv::Bool = save_csv,
    save_jld2::Bool = true,
    sample_thin::Integer = 1,
    metadata = Dict(),
    settings = Dict(),
    posterior_summary = nothing,
)
    mkpath(output_dir)
    chains = result_to_chains(result)
    wide = _thin_wide_samples(chains_to_wide_dataframe(chains), sample_thin)
    meta = _base_metadata(result; run_id = run_id, metadata = metadata, settings = settings)
    meta["saved_sample_thin"] = sample_thin
    diagnostics = sampling_diagnostics_dict(result.diagnostics)
    summary_df = posterior_summary === nothing ? _posterior_summary_dataframe(chains) : DataFrame(posterior_summary)

    if save_wide_csv
        _write_csv(joinpath(output_dir, "samples_wide.csv"), wide)
    end
    if save_csv
        _write_csv(joinpath(output_dir, "posterior_summary.csv"), summary_df)
    end
    _write_json(joinpath(output_dir, "metadata.json"), meta)
    _write_json(joinpath(output_dir, "diagnostics.json"), merge(copy(diagnostics), _sampler_acceptance_metadata(result)))

    if save_jld2
        JLD2 = _require_pkg(:JLD2)
        saved_chains = sample_thin == 1 ? chains : dataframe_to_chains(wide)
        bundle = Dict(
            "samples" => wide,
            "samples_chain" => saved_chains,
            "diagnostics" => diagnostics,
            "metadata" => meta,
            "settings" => Dict(String(k) => _jsonable(v) for (k, v) in pairs(settings)),
            "data_info" => Dict(
                "n" => length(result.data.time),
                "p" => size(result.data.z, 2),
                "number_of_events" => count(!=(0), result.data.event),
                "transform" => String(result.data.transform),
                "pseudocount" => result.data.pseudocount,
                "closure" => result.data.closure,
                "standardize" => result.data.standardize,
            ),
            "posterior_summary" => summary_df,
        )
        _call_latest(getproperty(JLD2, :jldsave), joinpath(output_dir, "result_bundle.jld2"), IOStream; result_bundle = bundle)
    end

    return (
        output_dir = output_dir,
        samples_wide = joinpath(output_dir, "samples_wide.csv"),
        result_bundle = joinpath(output_dir, "result_bundle.jld2"),
        metadata = joinpath(output_dir, "metadata.json"),
        diagnostics = joinpath(output_dir, "diagnostics.json"),
        posterior_summary = joinpath(output_dir, "posterior_summary.csv"),
    )
end

function _load_chains_from_run(input_dir::AbstractString)
    bundle_path = joinpath(input_dir, "result_bundle.jld2")
    if isfile(bundle_path)
        try
            JLD2 = _require_pkg(:JLD2)
            bundle = _call_latest(getproperty(JLD2, :load), bundle_path, "result_bundle")
            if haskey(bundle, "samples_chain")
                return bundle["samples_chain"]
            elseif haskey(bundle, "samples")
                return dataframe_to_chains(bundle["samples"])
            end
        catch err
            @warn "Could not read Chains from result_bundle.jld2; falling back to samples_wide.csv." exception=(err, catch_backtrace())
        end
    end
    wide_path = joinpath(input_dir, "samples_wide.csv")
    isfile(wide_path) || throw(ArgumentError("no result_bundle.jld2 or samples_wide.csv found in $input_dir"))
    return dataframe_to_chains(CSV.read(wide_path, DataFrame))
end

function _read_metadata(input_dir::AbstractString)
    path = joinpath(input_dir, "metadata.json")
    isfile(path) || return Dict{String,Any}()
    try
        JSON = _require_pkg(:JSON)
        return Dict{String,Any}(_call_latest(getproperty(JSON, :parsefile), path))
    catch
        return Dict{String,Any}()
    end
end

function _selected_params(chains; params = nothing, top_k_beta::Integer = 10)
    all_names = string.(names(chains))
    if params !== nothing
        raw = params isa AbstractString ? String.(strip.(split(params, ","))) : string.(params)
        return [p for p in raw if p in all_names]
    end
    selected = String[]
    "logposterior" in all_names && push!(selected, "logposterior")
    beta_names = [n for n in all_names if occursin(r"^beta\[\d+\]$", n)]
    if !isempty(beta_names) && top_k_beta > 0
        arr, param_names = _chains_array_and_names(chains)
        means = Dict(param_names[j] => mean(vec(arr[:, j, :])) for j in eachindex(param_names))
        ordered = sort(beta_names; by = n -> -abs(get(means, n, 0.0)))
        append!(selected, first(ordered, min(top_k_beta, length(ordered))))
    end
    "theta" in all_names && push!(selected, "theta")
    return unique(selected)
end

function _subset_chains(chains, params::AbstractVector{<:AbstractString})
    isempty(params) && return chains
    return chains[:, Symbol.(params), :]
end

function _save_plot_all_formats(plot_obj, output_dir::AbstractString, stem::AbstractString, formats)
    _configure_headless_gr!()
    Plots = _require_pkg(:Plots)
    mkpath(output_dir)
    fmt_list = formats isa AbstractString ? strip.(split(formats, ",")) : string.(formats)
    paths = String[]
    for fmt in fmt_list
        path = joinpath(output_dir, "$stem.$fmt")
        if lowercase(fmt) == "eps"
            open(path, "w") do io
                _call_latest(show, io, MIME("application/postscript"), plot_obj)
            end
        else
            _call_latest(getproperty(Plots, :savefig), plot_obj, path)
        end
        push!(paths, path)
    end
    return paths
end

function _plot_mcmc_chains(chains, output_dir::AbstractString, stem::AbstractString, seriestype::Symbol; formats, kwargs...)
    _configure_headless_gr!()
    _require_pkg(:StatsPlots)
    Plots = _require_pkg(:Plots)
    try
        _call_latest(getproperty(Plots, :gr))
    catch
    end
    plot_obj = _call_latest(getproperty(Plots, :plot), chains; seriestype = seriestype, kwargs...)
    return _save_plot_all_formats(plot_obj, output_dir, stem, formats)
end

function _rhat_dataframe(summary::DataFrame)
    (:parameters in propertynames(summary) && :rhat in propertynames(summary)) || return DataFrame()
    rows = summary[.!ismissing.(summary.rhat), [:parameters, :rhat]]
    rows = rows[isfinite.(Float64.(rows.rhat)), :]
    return rows
end

function _beta_rhat_dataframe(summary::DataFrame)
    rows = _rhat_dataframe(summary)
    isempty(rows) && return rows
    keep = occursin.(r"^beta\[\d+\]$", string.(rows.parameters))
    rows = rows[keep, :]
    rows.beta_index = [parse(Int, match(r"\[(\d+)\]", string(p)).captures[1]) for p in rows.parameters]
    sort!(rows, :beta_index)
    return rows
end

function _save_rhat_plots(summary::DataFrame, output_dir::AbstractString; rhat_threshold::Real, formats)
    _configure_headless_gr!()
    Plots = _require_pkg(:Plots)
    try
        _call_latest(getproperty(Plots, :gr))
    catch
    end
    paths = String[]
    beta_rows = _beta_rhat_dataframe(summary)
    if nrow(beta_rows) > 0
        p = _call_latest(getproperty(Plots, :scatter),
            beta_rows.beta_index,
            Float64.(beta_rows.rhat);
            xlabel = "beta index",
            ylabel = "R-hat",
            label = "beta",
            title = "R-hat by beta index",
        )
        _call_latest(getproperty(Plots, :hline!), p, [Float64(rhat_threshold)]; label = "threshold")
        append!(paths, _save_plot_all_formats(p, output_dir, "rhat_scatter", formats))
    end
    all_rows = _rhat_dataframe(summary)
    if nrow(all_rows) > 0
        p = _call_latest(getproperty(Plots, :histogram),
            Float64.(all_rows.rhat);
            xlabel = "R-hat",
            ylabel = "count",
            label = "parameters",
            title = "R-hat histogram",
        )
        _call_latest(getproperty(Plots, :vline!), p, [Float64(rhat_threshold)]; label = "threshold")
        append!(paths, _save_plot_all_formats(p, output_dir, "rhat_histogram", formats))
    end
    return paths
end

function _gamma_inclusion_dataframe(chains)
    arr, param_names = _chains_array_and_names(chains)
    gamma_idx = findall(n -> occursin(r"^gamma\[\d+\]$", n), param_names)
    isempty(gamma_idx) && return DataFrame()
    return DataFrame(
        parameter = param_names[gamma_idx],
        inclusion_probability = [mean(vec(arr[:, j, :])) for j in gamma_idx],
    )
end

function save_diagnostic_outputs(
    chains,
    output_dir::AbstractString;
    metadata = Dict(),
    params = nothing,
    top_k_beta::Integer = 10,
    max_lag::Integer = 50,
    rhat_threshold::Real = 1.01,
    formats = "png,pdf",
    plots::Bool = true,
)
    mkpath(output_dir)
    selected = _selected_params(chains; params = params, top_k_beta = top_k_beta)

    summary = _diagnostics_summary_dataframe(
        chains;
        metadata = metadata,
        rhat_threshold = rhat_threshold,
    )
    _write_csv(joinpath(output_dir, "diagnostics_summary.csv"), summary)
    _write_json(joinpath(output_dir, "diagnostics_summary.json"), [Dict(String(c) => _jsonable(row[c]) for c in propertynames(summary)) for row in eachrow(summary)])

    gamma_df = _gamma_inclusion_dataframe(chains)
    !isempty(gamma_df) && _write_csv(joinpath(output_dir, "gamma_inclusion_probabilities.csv"), gamma_df)

    plot_paths = String[]
    if plots
        plot_chains = _subset_chains(chains, selected)
        if !isempty(selected)
            append!(plot_paths, _plot_mcmc_chains(plot_chains, output_dir, "traceplot", :traceplot; formats = formats))
            append!(plot_paths, _plot_mcmc_chains(plot_chains, output_dir, "density", :mixeddensity; formats = formats))
            append!(plot_paths, _plot_mcmc_chains(plot_chains, output_dir, "acf", :autocorplot; formats = formats, maxlag = max_lag))
        end

        if size(chains, 3) <= 1
            @warn "R-hat requires multiple chains; R-hat plots were skipped."
        else
            append!(plot_paths, _save_rhat_plots(summary, output_dir; rhat_threshold = rhat_threshold, formats = formats))
        end
    end

    return (
        output_dir = output_dir,
        selected_params = selected,
        diagnostics_summary = joinpath(output_dir, "diagnostics_summary.csv"),
        plot_paths = plot_paths,
    )
end

function save_diagnostic_outputs(
    result::CoxSamplerResult;
    output_dir::AbstractString,
    metadata = Dict(),
    params = nothing,
    top_k_beta::Integer = 10,
    max_lag::Integer = 50,
    rhat_threshold::Real = 1.01,
    formats = "png,pdf",
    plots::Bool = true,
)
    return save_diagnostic_outputs(
        result_to_chains(result),
        output_dir;
        metadata = metadata,
        params = params,
        top_k_beta = top_k_beta,
        max_lag = max_lag,
        rhat_threshold = rhat_threshold,
        formats = formats,
        plots = plots,
    )
end

function diagnose_run(
    input_dir::AbstractString;
    output_dir::AbstractString = joinpath(input_dir, "diagnose"),
    params = nothing,
    top_k_beta::Integer = 10,
    max_lag::Integer = 50,
    rhat_threshold::Real = 1.01,
    formats = "png,pdf",
    plots::Bool = true,
)
    mkpath(output_dir)
    chains = _load_chains_from_run(input_dir)
    metadata = _read_metadata(input_dir)
    return save_diagnostic_outputs(
        chains,
        output_dir;
        metadata = metadata,
        params = params,
        top_k_beta = top_k_beta,
        max_lag = max_lag,
        rhat_threshold = rhat_threshold,
        formats = formats,
        plots = plots,
    )
end
