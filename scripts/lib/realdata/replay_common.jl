# Replay saved real-data inputs and fitting settings. Caller loads CSV/DataFrames/JSON.
const PRIOR_KEYS = [
    "hs_tau_scale", "hs_local_scale", "rhs_tau_scale", "rhs_local_scale",
    "rhs_slab_df", "rhs_slab_s2", "rhs_pseudo_sigma_df", "rhs_pseudo_sigma_scale",
    "lasso_lambda", "lasso_lambda2_shape", "lasso_lambda2_rate",
    "sslasso_spike_scale", "sslasso_slab_scale", "sslasso_theta_a", "sslasso_theta_b",
    "prox_l1_lambda", "alpha_prior_shape", "alpha_prior_scale", "alpha_prior_kind",
    "prox_sum0_lambda", "prox_root_tol", "prox_root_maxiter",
]
const INTEGER_PRIOR_KEYS = Set(["prox_root_maxiter"])

function parse_args(args)
    opts = Dict{String,String}()
    i = 1
    while i <= length(args)
        key = args[i]
        startswith(key, "--") || throw(ArgumentError("unexpected argument: $key"))
        name = key[3:end]
        if i == length(args) || startswith(args[i + 1], "--")
            opts[name] = "true"
            i += 1
        else
            opts[name] = args[i + 1]
            i += 2
        end
    end
    return opts
end

opt(opts, name, default) = get(opts, name, default)

function maybe_float(x)
    x === missing && return missing
    x === nothing && return missing
    x isa Number && return Float64(x)
    s = strip(string(x))
    isempty(s) && return missing
    lowercase(s) in ("missing", "nothing", "nan", "na") && return missing
    try
        return parse(Float64, s)
    catch
        return missing
    end
end

function maybe_int(x)
    v = maybe_float(x)
    v === missing && return missing
    return Int(round(v))
end

function maybe_bool(x)
    x isa Bool && return x
    s = lowercase(strip(string(x)))
    return s in ("1", "true", "yes", "y")
end

function split_csv_string(x)
    s = strip(string(x))
    isempty(s) && return String[]
    return [strip(part) for part in split(s, ',') if !isempty(strip(part))]
end

function read_json(path::AbstractString)
    isfile(path) || throw(ArgumentError("metadata not found: $path"))
    return JSON.parsefile(path; use_mmap = false)
end

function input_path_from_metadata(meta)
    input = string(meta["input"])
    return isabspath(input) ? input : abspath(joinpath(@__DIR__, "..", "..", "..", input))
end

function drop_missing_survival_rows_local(df::DataFrame, time_col::Symbol, status_col::Symbol)
    mask = .!ismissing.(df[!, time_col]) .& .!ismissing.(df[!, status_col])
    return df[mask, :], count(.!mask)
end

function realdata_arrays(meta)
    input = input_path_from_metadata(meta)
    df = CSV.read(input, DataFrame)
    time_col = Symbol(get(meta, "time_col", "time"))
    status_col = Symbol(get(meta, "status_col", "status"))
    df, _ = drop_missing_survival_rows_local(df, time_col, status_col)
    component_cols = split_csv_string(get(meta, "component_cols", ""))
    isempty(component_cols) && throw(ArgumentError("metadata has no component_cols for $input"))
    X = Matrix{Float64}(coalesce.(Matrix(df[:, component_cols]), 0.0))
    time = Float64.(df[!, time_col])
    event = Int.(df[!, status_col])
    return X, time, event
end

function metadata_kwargs(meta; samples::Int, warmup::Int, chains::Int, seed::Int)
    kwargs = Dict{Symbol,Any}(
        :prior_name => string(meta["prior_name"]),
        :sampler => Symbol(get(meta, "sampler", "nuts")),
        :nuts_backend => Symbol(get(meta, "nuts_backend", "explicit")),
        :n_samples => samples,
        :warmup => warmup,
        :thin => Int(get(meta, "thin", 1)),
        :chains => chains,
        :multichains => maybe_bool(get(meta, "multichains", false)),
        :target_accept => Float64(get(meta, "target_accept", 0.9)),
        :max_depth => Int(get(meta, "max_depth", 10)),
        :hmc_n_leapfrog => Int(get(meta, "hmc_n_leapfrog", 10)),
        :hmc_step_size => get(meta, "hmc_step_size", nothing),
        :noncentered => maybe_bool(get(meta, "noncentered", false)),
        :seed => seed,
        :use_car => maybe_bool(get(meta, "use_car", false)),
        :standardize => maybe_bool(get(meta, "standardize", false)),
        :tie_method => Symbol(get(meta, "tie_method", "efron")),
        :p0 => get(meta, "p0", nothing),
        :p0_alpha => Float64(get(meta, "p0_alpha", 0.05)),
        :constraint => Symbol(get(meta, "constraint", "soft_center")),
        :prox_sum0_lambda => Float64(get(meta, "prox_sum0_lambda", 1e-6)),
        :progress => false,
        :approx => false,
        :approx_point => :zero,
        :realdata => true,
    )
    for key in PRIOR_KEYS
        haskey(meta, key) || continue
        val = meta[key]
        val === nothing && continue
        if key in ("alpha_prior_kind",)
            kwargs[Symbol(key)] = Symbol(val)
        elseif key in INTEGER_PRIOR_KEYS
            parsed = maybe_int(val)
            parsed === missing || (kwargs[Symbol(key)] = parsed)
        else
            parsed = maybe_float(val)
            parsed === missing || (kwargs[Symbol(key)] = parsed)
        end
    end
    # Preserve optional sampler/CAR settings when replaying saved configurations.
    for key in ("prox_block_step_size", "prox_block_min_step_size", "prox_block_max_step_size",
                "prox_block_divergence_threshold", "car_weight_quantile", "car_precision",
                "car_strength_prior_a", "car_strength_prior_b", "car_kappa_init")
        value = get(meta, key, nothing)
        value === nothing || (kwargs[Symbol(key)] = Float64(value))
    end
    for key in ("prox_block_metric", "car_distance", "car_degree_scale", "init")
        value = get(meta, key, nothing)
        value === nothing || (kwargs[Symbol(key)] = Symbol(value))
    end
    haskey(meta, "prox_block_adapt_every") &&
        (kwargs[:prox_block_adapt_every] = Int(meta["prox_block_adapt_every"]))
    haskey(meta, "sample_car_strength") &&
        (kwargs[:sample_car_strength] = maybe_bool(meta["sample_car_strength"]))
    return kwargs
end

