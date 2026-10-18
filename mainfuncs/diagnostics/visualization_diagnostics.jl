function _configure_headless_gr!()
    if !haskey(ENV, "GKSwstype") && !haskey(ENV, "DISPLAY")
        ENV["GKSwstype"] = "100"
    end
    return nothing
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

_plot_margin(Plots, value::Real) = _call_latest(*, value, getproperty(Plots, :mm))

function _plot_mcmc_chains(chains, output_dir::AbstractString, stem::AbstractString, seriestype::Symbol; formats, kwargs...)
    _configure_headless_gr!()
    _require_pkg(:StatsPlots)
    Plots = _require_pkg(:Plots)
    try
        _call_latest(getproperty(Plots, :gr))
    catch
    end
    n_params = max(size(chains, 2), 1)
    plot_kwargs = merge(
        (
            seriestype = seriestype,
            size = (1050, max(680, 150 * n_params)),
            left_margin = _plot_margin(Plots, 5),
            right_margin = _plot_margin(Plots, 3),
            bottom_margin = _plot_margin(Plots, 4),
            top_margin = _plot_margin(Plots, 4),
            tickfontsize = 7,
            guidefontsize = 9,
            titlefontsize = 10,
        ),
        NamedTuple(kwargs),
    )
    plot_obj = _call_latest(getproperty(Plots, :plot), chains; plot_kwargs...)
    return _save_plot_all_formats(plot_obj, output_dir, stem, formats)
end


function _convergence_scatter(summary::AbstractDataFrame, metric::Symbol, n_chains::Integer)
    n_chains > 0 || throw(ArgumentError("n_chains must be positive"))
    metric in (:rhat, :ess_bulk, :ess_tail) || throw(ArgumentError("unsupported convergence metric"))
    _configure_headless_gr!()
    Plots = _require_pkg(:Plots)
    rows = :category in propertynames(summary) ? summary[summary.category .== "convergence", :] : summary
    valid = [x isa Real && isfinite(x) for x in rows[!, metric]]
    indices = findall(valid)
    values = Float64.(rows[valid, metric])
    labels = string.(rows.parameters[valid])
    ylabel = metric == :rhat ? "R-hat" : metric == :ess_bulk ? "Bulk ESS" : "Tail ESS"
    p = _call_latest(getproperty(Plots, :scatter), indices, values;
        xlabel = "Parameter", ylabel, label = ylabel, title = "$ylabel by parameter",
        xticks = length(indices) <= 25 ? (indices, labels) : :auto,
        xrotation = 45, markersize = 4, size = (1000, 560),
        bottom_margin = _plot_margin(Plots, 10), left_margin = _plot_margin(Plots, 6))
    thresholds = metric == :rhat ? [1.01, 1.1] : [Float64(100 * n_chains)]
    for threshold in thresholds
        label = metric == :rhat ? "R-hat = $threshold" : "ESS = $(Int(threshold)) (100 x $n_chains chains)"
        _call_latest(getproperty(Plots, :hline!), p, [threshold];
            color = :red, linestyle = :dash, linewidth = 1.5, label)
    end
    return p
end

function _save_convergence_plots(summary, output_dir; n_chains::Integer, formats)
    paths = String[]
    for metric in (:rhat, :ess_bulk, :ess_tail)
        metric == :rhat && n_chains == 1 && continue
        plot = _convergence_scatter(summary, metric, n_chains)
        append!(paths, _save_plot_all_formats(plot, output_dir, "$(metric)_scatter", formats))
    end
    return paths
end

"""Write the unified numerical table and optional trace, density, ACF and convergence plots."""
function save_diagnostic_outputs(chains, output_dir::AbstractString;
    params = nothing, top_k_beta::Integer = 10,
    max_lag::Integer = 50, rhat_threshold::Real = 1.01, formats = "pdf",
    plots::Bool = true, transitions = DataFrame(), runtime = Dict(), summary = nothing)
    mkpath(output_dir)
    selected = _selected_params(chains; params, top_k_beta)
    table = summary === nothing ? diagnostic_table(chains_to_wide_dataframe(chains);
        transitions, runtime, rhat_threshold) : copy(summary)
    convergence = table.category .== "convergence"
    table[convergence, :rhat_threshold] .= Float64(rhat_threshold)
    _write_csv(joinpath(output_dir, "diagnostics_summary.csv"), table)
    isempty(transitions) || _write_csv(joinpath(output_dir, "transition_diagnostics.csv"), transitions)
    paths = String[]
    if plots
        if !isempty(selected)
            subset = _subset_chains(chains, selected)
            append!(paths, _plot_mcmc_chains(subset, output_dir, "traceplot", :traceplot; formats))
            append!(paths, _plot_mcmc_chains(subset, output_dir, "density", :mixeddensity; formats))
            append!(paths, _plot_mcmc_chains(subset, output_dir, "acf", :autocorplot; formats, maxlag = max_lag))
        end
        # Saved numerical diagnostics can use more draws than a thinned plot bundle.
        counts = collect(skipmissing(table[convergence, :n_chains]))
        n_chains = isempty(counts) ? size(chains, 3) : Int(first(counts))
        append!(paths, _save_convergence_plots(table, output_dir; n_chains, formats))
    end
    return (output_dir = output_dir, selected_params = selected,
        diagnostics_summary = joinpath(output_dir, "diagnostics_summary.csv"), plot_paths = paths)
end

function save_diagnostic_outputs(result::CoxSamplerResult; output_dir::AbstractString, kwargs...)
    wide = samples_wide_dataframe(result)
    _unstandardize_wide_beta!(wide, result.data)
    return save_diagnostic_outputs(dataframe_to_chains(wide), output_dir;
        transitions = _result_transitions(result),
        runtime = merge(sampling_diagnostics_dict(result.diagnostics), _sampler_acceptance_metadata(result)), kwargs...)
end

function diagnose_run(input_dir::AbstractString; output_dir::AbstractString = input_dir, kwargs...)
    chains = _load_chains_from_run(input_dir)
    numerical_path = joinpath(input_dir, "diagnostics_summary.csv")
    summary = isfile(numerical_path) ? read_diagnostics(input_dir) : nothing
    transition_path = joinpath(input_dir, "transition_diagnostics.csv")
    transitions = isfile(transition_path) ? CSV.read(transition_path, DataFrame) : DataFrame()
    return save_diagnostic_outputs(chains, output_dir; summary, transitions, kwargs...)
end
