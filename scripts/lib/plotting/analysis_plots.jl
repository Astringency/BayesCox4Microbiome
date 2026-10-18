function _plot_path_with_ext(path::AbstractString, ext::AbstractString)
    root, _ = splitext(path)
    return root * ext
end

function _save_plots_vector(plot_obj, path::AbstractString; save_eps::Bool = true)
    pdf_path = _plot_path_with_ext(path, ".pdf")
    ensure_dir(dirname(pdf_path))
    Plots.savefig(plot_obj, pdf_path)
    if save_eps
        eps_path = _plot_path_with_ext(path, ".eps")
        open(eps_path, "w") do io
            Base.invokelatest(show, io, MIME("application/postscript"), plot_obj)
        end
    end
    return pdf_path
end

function _coefficient_diagnostic_title(summary::DataFrame, estimate::Symbol, estimates::AbstractVector, truth)
    truth === nothing && return ""
    truthv = Float64.(truth)
    length(truthv) == length(estimates) ||
        throw(DimensionMismatch("truth length must match summary row count."))
    mse = coefficient_mse(truthv, Float64.(estimates))
    fd = false_discovery(truthv, summary)
    estimate_name = estimate == :map ? "MAP" : String(estimate)
    return @sprintf(
        "MSE(%s)=%.4f | FDR=%.4f (FP=%d, FN=%d, TP=%d, TN=%d)",
        estimate_name,
        mse,
        fd.FDR,
        fd.FP,
        fd.FN,
        fd.TP,
        fd.TN,
    )
end

function coefficient_plot(
    summary::DataFrame,
    path::AbstractString;
    truth = nothing,
    title = "Coefficient estimates",
    estimate::Symbol = :mean,
)
    x = 1:nrow(summary)
    estimate in propertynames(summary) ||
        throw(ArgumentError("summary does not contain estimate column $estimate."))
    (:lower in propertynames(summary) && :upper in propertynames(summary)) ||
        throw(ArgumentError("summary must contain lower and upper columns."))
    y = Float64.(summary[!, estimate])
    lower = Float64.(summary.lower)
    upper = Float64.(summary.upper)
    estimate_label = estimate == :map ? "MAP" : "posterior $(String(estimate))"
    diagnostic_title = _coefficient_diagnostic_title(summary, estimate, y, truth)
    plot_title = isempty(diagnostic_title) ? title : "$title\n$diagnostic_title"
    y_values = vcat(lower, upper, y, [0.0], truth === nothing ? Float64[] : Float64.(truth))
    ymin, ymax = extrema(y_values)
    pad = max(0.12 * (ymax - ymin), 0.15)
    ymin -= pad
    ymax += pad
    try
        Plots.gr()
    catch
    end
    p = Plots.scatter(
        x,
        y;
        yerror = (y .- lower, upper .- y),
        label = estimate_label,
        xlabel = "Taxon index",
        ylabel = "Coefficient",
        title = plot_title,
        ylims = (ymin, ymax),
        color = :steelblue,
        markerstrokecolor = :steelblue,
        markersize = 3,
        linewidth = 1.2,
        legend = :topright,
        grid = true,
        size = (1000, 520),
        titlefontsize = 10,
        guidefontsize = 10,
        tickfontsize = 8,
        legendfontsize = 8,
        left_margin = 9Plots.mm,
        right_margin = 7Plots.mm,
        bottom_margin = 9Plots.mm,
        top_margin = 9Plots.mm,
    )
    Plots.hline!(p, [0.0]; label = "", color = :gray, linestyle = :dash, linewidth = 1)
    if truth !== nothing
        truthv = Float64.(truth)
        length(truthv) == nrow(summary) ||
            throw(DimensionMismatch("truth length must match summary row count."))
        Plots.scatter!(
            p,
            x,
            truthv;
            label = "true",
            color = :firebrick,
            markerstrokecolor = :firebrick,
            markersize = 3,
        )
    end
    return _save_plots_vector(p, path)
end

function _finite_mean(v)
    vals = Float64[]
    for x in skipmissing(v)
        xf = Float64(x)
        isfinite(xf) && push!(vals, xf)
    end
    return isempty(vals) ? NaN : mean(vals)
end

function _metric_values(rows::DataFrame, metric::Symbol)
    agg = combine(groupby(rows, [:prior, :estimate]), metric => _finite_mean => :value)
    priors = sort(unique(agg.prior))
    estimate_order = ["mean", "median", "MAP"]
    estimates = sort(unique(agg.estimate); by = e -> begin
        pos = findfirst(==(string(e)), estimate_order)
        pos === nothing ? length(estimate_order) + 1 : pos
    end)
    return agg, priors, estimates
end

function _metric_panel_plot(rows::DataFrame, metric::Symbol; title::AbstractString)
    agg, priors, estimates = _metric_values(rows, metric)
    p = Plots.plot(;
        title = title,
        xlabel = "",
        ylabel = "",
        legend = false,
        grid = true,
        gridalpha = 0.22,
        framestyle = :box,
        titlefontsize = 10,
        tickfontsize = 7,
        guidefontsize = 8,
        left_margin = 6Plots.mm,
        right_margin = 3Plots.mm,
        bottom_margin = 7Plots.mm,
        top_margin = 5Plots.mm,
    )
    isempty(priors) && return p
    vals = [Float64(v) for v in agg.value if isfinite(Float64(v))]
    isempty(vals) && return p
    xbase = collect(1:length(priors))
    n_est = max(length(estimates), 1)
    bar_w = min(0.72 / n_est, 0.24)
    offsets = collect(range(-bar_w * (n_est - 1) / 2, bar_w * (n_est - 1) / 2; length = n_est))
    palette = ["#2F5D8C", "#5B8FC1", "#A9CBE8", "#6BAF92", "#B8D8BA"]
    all_label_y = Float64[]
    for (k, est) in enumerate(estimates)
        xs = Float64[]
        ys = Float64[]
        for (i, prior) in enumerate(priors)
            idx = findfirst((agg.prior .== prior) .& (agg.estimate .== est))
            idx === nothing && continue
            value = agg.value[idx]
            isfinite(value) || continue
            push!(xs, xbase[i] + offsets[k])
            push!(ys, Float64(value))
        end
        isempty(xs) && continue
        Plots.bar!(
            p,
            xs,
            ys;
            bar_width = bar_w,
            label = string(est),
            color = palette[mod1(k, length(palette))],
            linecolor = :white,
            linewidth = 0.6,
            fillalpha = 0.92,
        )
        span = max(maximum(vals) - minimum(vals), 1e-6)
        offset = max(0.025 * span, 1e-4)
        for (x, y) in zip(xs, ys)
            label_y = y >= 0 ? y + offset : y - offset
            push!(all_label_y, label_y)
            valign = y >= 0 ? :bottom : :top
            Plots.annotate!(p, x, label_y, Plots.text(@sprintf("%.2f", y), 6, :center, valign))
        end
    end
    y_candidates = vcat(vals, [0.0], all_label_y)
    is_cindex = startswith(string(metric), "Cindex")
    ymin = is_cindex ? 0.0 : minimum(y_candidates)
    ymax = is_cindex ? max(1.0, maximum(y_candidates)) : maximum(y_candidates)
    span = max(ymax - ymin, 1e-6)
    ymin -= is_cindex ? 0.0 : 0.12 * span
    ymax += 0.18 * span
    labels = [length(string(prior)) > 10 ? string(prior)[1:10] : string(prior) for prior in priors]
    Plots.plot!(
        p;
        xticks = (xbase, labels),
        xrotation = 35,
        ylims = (ymin, ymax),
        legend = :topright,
    )
    return p
end

function metrics_plot(metrics::DataFrame, path::AbstractString; title = "Model metrics")
    plot_rows = copy(metrics)
    plot_rows.prior = string.(plot_rows.prior)
    plot_rows.estimate = string.(plot_rows.estimate)
    for col in (:prior, :estimate, :WAIC_pw1, :LOOIC, :NLPML, :Cindex)
        col in propertynames(plot_rows) ||
            throw(ArgumentError("metrics table does not contain required column $col."))
    end
    try
        Plots.gr()
    catch
    end
    panels = (
        _metric_panel_plot(plot_rows, :WAIC_pw1; title = "WAIC"),
        _metric_panel_plot(plot_rows, :LOOIC; title = "LOOIC"),
        _metric_panel_plot(plot_rows, :NLPML; title = "NLPML"),
        _metric_panel_plot(plot_rows, :Cindex; title = "Training C-index"),
    )
    has_test = :Cindex_test in propertynames(plot_rows)
    has_test && (panels = (panels...,
        _metric_panel_plot(plot_rows, :Cindex_test; title = "Test C-index (n=1000)")))
    p = Plots.plot(
        panels...;
        layout = has_test ? (3, 2) : (2, 2),
        plot_title = title,
        size = (1200, 850),
        plot_titlefontsize = 13,
    )
    return _save_plots_vector(p, path)
end


function write_simulation_plots(
    run_dir::AbstractString,
    figs_dir::AbstractString,
    summary::DataFrame,
    sim;
    label::AbstractString,
    prior::AbstractString,
    baseline,
    n::Integer,
    p::Integer,
    nonzero::Integer,
)
    coef_paths = [
        joinpath(figs_dir, "coef_$label.pdf"),
        joinpath(figs_dir, "coef_median_$label.pdf"),
        joinpath(figs_dir, "coef_map_$label.pdf"),
    ]
    coefficient_plot(
        summary,
        coef_paths[1];
        truth = sim.beta,
        title = "Simulation $prior $baseline n=$n p=$p nonzero=$nonzero",
        estimate = :mean,
    )
    coefficient_plot(
        summary,
        coef_paths[2];
        truth = sim.beta,
        title = "Simulation $prior median $baseline n=$n p=$p nonzero=$nonzero",
        estimate = :median,
    )
    coefficient_plot(
        summary,
        coef_paths[3];
        truth = sim.beta,
        title = "Simulation $prior MAP $baseline n=$n p=$p nonzero=$nonzero",
        estimate = :map,
    )
    return nothing
end

