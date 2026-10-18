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

_call_latest(f, args...; kwargs...) = Base.invokelatest(f, args...; kwargs...)

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
        "tie_method" => String(data.tie_method),
        "zero_replacement" => data.zero_replacement,
        "zero_handling" => "half_minimum_positive",
        "runtime" => result.diagnostics.total_seconds,
        "git_commit_hash" => git_sha,
        "git_sha" => git_sha,
        "created_at" => string(now()),
    )
    merge!(base, _sampler_acceptance_metadata(result))
    # Preserve all supplied settings even when JLD2 export is disabled.
    merge!(base, Dict(String(k) => _safe_string(v) for (k, v) in pairs(settings)))
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

function _beta_column_index(col)
    m = match(r"^beta\[(\d+)\]$", string(col))
    m === nothing && return nothing
    return parse(Int, m.captures[1])
end

function _unstandardize_wide_beta!(wide::DataFrame, data::CoxPreparedData)
    data.standardize || return wide
    for col in propertynames(wide)
        j = _beta_column_index(col)
        j === nothing && continue
        1 <= j <= length(data.col_scale) || continue
        wide[!, col] = Float64.(wide[!, col]) ./ data.col_scale[j]
    end
    return wide
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
            if sc == "2.5%"
                rename_map[c] = :q025
            elseif sc in ("50.0%", "50%", "median")
                rename_map[c] = :median
            elseif sc == "97.5%"
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

"""Rank-normalized split/folded R-hat, bulk/tail ESS and posterior summaries.
Constant parameters do not demonstrate mixing; their R-hat and ESS are missing.
Single-chain R-hat is withheld. ESS thresholds use 100 times the chain count.
"""
function convergence_diagnostics(chains; rhat_threshold::Real = 1.01)
    df = _posterior_summary_dataframe(chains)
    :std in propertynames(df) && rename!(df, :std => :sd)
    df.parameters = string.(df.parameters)
    arr, params = _chains_array_and_names(chains)
    n_chains = size(arr, 3)
    df.constant_all_draws = [length(unique(vec(arr[:, findfirst(==(p), params), :]))) == 1 for p in df.parameters]
    for col in (:rhat, :ess_bulk, :ess_tail)
        col in propertynames(df) || (df[!, col] = fill(missing, nrow(df)))
        allowmissing!(df, col)
        df[df.constant_all_draws, col] .= missing
    end
    n_chains == 1 && (df.rhat .= missing)
    df.n_chains = fill(n_chains, nrow(df))
    df.n_draws = fill(size(arr, 1), nrow(df))
    df.rhat_threshold = fill(Float64(rhat_threshold), nrow(df))
    df.ess_threshold = fill(100 * n_chains, nrow(df))
    return df
end
convergence_diagnostics(wide::AbstractDataFrame; kwargs...) =
    convergence_diagnostics(dataframe_to_chains(DataFrame(wide)); kwargs...)

"""Combine parameter, constraint, mixing, transition and cost summaries in one table.
`category`, `parameters`, `chain` and `phase` identify the row's scope.
Raw transition records have a different grain and are saved separately.
"""
function diagnostic_table(wide::AbstractDataFrame; transitions = DataFrame(),
    runtime = Dict(), rhat_threshold::Real = 1.01)
    tables = DataFrame[]
    function add!(table, category; parameter = "", phase = "sample")
        isempty(table) && return
        table = copy(table)
        :parameter in propertynames(table) && rename!(table, :parameter => :parameters)
        :parameters in propertynames(table) || (table.parameters = fill(parameter, nrow(table)))
        :chain in propertynames(table) || (table.chain = fill("pooled", nrow(table)))
        table.chain = string.(table.chain)
        :phase in propertynames(table) || (table.phase = fill(phase, nrow(table)))
        table.category = fill(category, nrow(table))
        push!(tables, table)
    end
    add!(convergence_diagnostics(wide; rhat_threshold), "convergence")
    add!(zero_sum_diagnostics(wide), "zero_sum"; parameter = "sum(beta)")
    add!(selection_mixing_diagnostics(wide), "selection_mixing")
    add!(transition_diagnostics_summary(transitions), "transition")
    costs = runtime isa SamplingDiagnostics ? sampling_diagnostics_dict(runtime) : runtime
    isempty(costs) || add!(DataFrame([Symbol(k) => [_csv_value(v)] for (k,v) in pairs(costs)]), "runtime"; phase = "all")
    table = vcat(tables...; cols = :union)
    select!(table, :category, :parameters, :chain, :phase, Not([:category, :parameters, :chain, :phase]))
    return table
end

"""Read one category from the unified numerical diagnostic output."""
function read_diagnostics(run_dir::AbstractString; category = nothing)
    table = CSV.read(joinpath(run_dir, "diagnostics_summary.csv"), DataFrame)
    category === nothing && return table
    return table[table.category .== string(category), :]
end

function _save_numerical_diagnostics(wide, output_dir; transitions = DataFrame(), kwargs...)
    mkpath(output_dir)
    table = diagnostic_table(wide; transitions, kwargs...)
    _write_csv(joinpath(output_dir, "diagnostics_summary.csv"), table)
    isempty(transitions) || _write_csv(joinpath(output_dir, "transition_diagnostics.csv"), transitions)
    return table
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
    wide_full = chains_to_wide_dataframe(chains)
    _unstandardize_wide_beta!(wide_full, result.data)
    wide = _thin_wide_samples(wide_full, sample_thin)
    meta = _base_metadata(result; run_id = run_id, metadata = metadata, settings = settings)
    meta["saved_sample_thin"] = sample_thin
    meta["posterior_beta_scale"] = result.data.standardize ? "unstandardized" : "model"
    diagnostics = Dict{String,Any}(sampling_diagnostics_dict(result.diagnostics))
    transitions = _result_transitions(result)
    diagnostics["transition_diagnostics_available"] = !isempty(transitions)
    diagnostics["transition_diagnostics_definition"] = "numerical error or absolute Hamiltonian energy error > 1000; block samplers use their configured divergence threshold"
    output_chains = result.data.standardize || sample_thin != 1 ? dataframe_to_chains(wide) : chains
    summary_df = posterior_summary === nothing ? select(_posterior_summary_dataframe(output_chains),
        Not([:mcse, :ess_bulk, :ess_tail, :rhat, :ess_per_sec])) : DataFrame(posterior_summary)

    if save_wide_csv
        _write_csv(joinpath(output_dir, "samples_wide.csv"), wide)
    end
    if save_csv
        _write_csv(joinpath(output_dir, "posterior_summary.csv"), summary_df)
        _save_numerical_diagnostics(wide_full, output_dir; transitions,
            runtime = merge(sampling_diagnostics_dict(result.diagnostics), _sampler_acceptance_metadata(result)))
    end
    _write_json(joinpath(output_dir, "metadata.json"), meta)

    if save_jld2
        JLD2 = _require_pkg(:JLD2)
        bundle = Dict(
            "samples" => wide,
            "samples_chain" => output_chains,
            "diagnostics" => diagnostics,
            "transition_diagnostics" => transitions,
            "metadata" => meta,
            "settings" => Dict(String(k) => _jsonable(v) for (k, v) in pairs(settings)),
            "data_info" => Dict(
                "n" => length(result.data.time),
                "p" => size(result.data.z, 2),
                "number_of_events" => count(!=(0), result.data.event),
                "transform" => String(result.data.transform),
                "zero_replacement" => result.data.zero_replacement,
                "closure" => result.data.closure,
                "standardize" => result.data.standardize,
                "tie_method" => String(result.data.tie_method),
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
        diagnostics = joinpath(output_dir, "diagnostics_summary.csv"),
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

function _numeric_col_values(df::DataFrame, col::Symbol)
    vals = Float64[]
    col in propertynames(df) || return vals
    for x in df[!, col]
        x isa Real && isfinite(x) && push!(vals, Float64(x))
    end
    return vals
end

function _chain_diagnostics(result)
    chains = try
        result_to_chains(result)
    catch
        return Dict{Symbol,Any}(
            :rhat_max => missing,
            :rhat_median => missing,
            :ess_bulk_min => missing,
            :ess_bulk_median => missing,
            :ess_tail_min => missing,
            :ess_tail_median => missing,
        )
    end
    stats = try
        convergence_diagnostics(chains)
    catch
        DataFrame()
    end
    isempty(stats) && return Dict{Symbol,Any}(
        :rhat_max => missing,
        :rhat_median => missing,
        :ess_bulk_min => missing,
        :ess_bulk_median => missing,
        :ess_tail_min => missing,
        :ess_tail_median => missing,
    )
    beta_rows = stats[occursin.(r"^beta\[\d+\]$", stats.parameters), :]
    ess = _numeric_col_values(beta_rows, :ess_bulk)
    tail = _numeric_col_values(beta_rows, :ess_tail)
    rhat = _numeric_col_values(beta_rows, :rhat)
    return Dict{Symbol,Any}(
        :rhat_max => isempty(rhat) ? missing : maximum(rhat),
        :rhat_median => isempty(rhat) ? missing : median(rhat),
        :ess_bulk_min => isempty(ess) ? missing : minimum(ess),
        :ess_bulk_median => isempty(ess) ? missing : median(ess),
        :ess_tail_min => isempty(tail) ? missing : minimum(tail),
        :ess_tail_median => isempty(tail) ? missing : median(tail),
    )
end
