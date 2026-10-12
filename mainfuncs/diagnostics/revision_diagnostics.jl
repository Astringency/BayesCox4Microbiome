"""Summaries of per-draw coefficient sums on the original coefficient scale."""
function zero_sum_diagnostics(wide::AbstractDataFrame)
    beta_cols = [c for c in propertynames(wide) if occursin(r"^beta\[\d+\]$", String(c))]
    isempty(beta_cols) && return DataFrame()
    rows = NamedTuple[]
    groups = vcat([(string(k.chain), DataFrame(g)) for (k,g) in pairs(groupby(wide, :chain))],
        [("pooled", DataFrame(wide))])
    for (label, g) in groups
        beta = Matrix{Float64}(g[:, beta_cols])
        all(isfinite, beta) || throw(ArgumentError("nonfinite coefficients in zero-sum diagnostics"))
        s = vec(sum(beta; dims = 2))
        rounding_tol = 100eps(Float64) .* max.(1.0, vec(sum(abs, beta; dims = 2)))
        q = quantile(s, [0.025, 0.25, 0.5, 0.75, 0.975])
        push!(rows, (chain = label, n_draws = length(s), p = size(beta, 2),
            mean = mean(s), sd = length(s) > 1 ? std(s) : missing,
            q025 = q[1], q25 = q[2], median = q[3], q75 = q[4], q975 = q[5],
            mean_abs = mean(abs, s), rms = sqrt(mean(abs2, s)), max_abs = maximum(abs, s),
            prob_abs_lt_1e_2 = mean(abs.(s) .< 1e-2),
            prob_abs_lt_1e_3 = mean(abs.(s) .< 1e-3),
            prob_abs_lt_1e_6 = mean(abs.(s) .< 1e-6),
            prob_abs_lt_1e_8 = mean(abs.(s) .< 1e-8),
            fraction_within_roundoff = mean(abs.(s) .<= rounding_tol),
            sum_posterior_means = sum(vec(mean(beta; dims = 1))),
            sum_marginal_medians = sum([median(view(beta, :, j)) for j in axes(beta, 2)])))
    end
    return DataFrame(rows)
end

function selection_mixing_diagnostics(wide::AbstractDataFrame)
    cols = [c for c in propertynames(wide) if occursin(r"^gamma\[\d+\]$", String(c))]
    rows = NamedTuple[]
    for g in groupby(wide, :chain), col in cols
        order = sortperm(g.iter)
        raw = Float64.(g[order, col])
        all(x -> x == 0 || x == 1, raw) || throw(ArgumentError("gamma draws must be binary"))
        x = raw .== 1
        d = diff(Int.(x))
        switches = count(!=(0), d)
        ends = vcat(0, findall(!=(0), d), length(x))
        lag1 = length(x) > 2 && std(raw[1:end-1]) > 0 && std(raw[2:end]) > 0 ?
            cor(raw[1:end-1], raw[2:end]) : missing
        push!(rows, (chain = string(first(g.chain)), parameter = String(col), n_draws = length(x),
            inclusion_probability = mean(x), n_switches = switches,
            switch_rate = length(x) > 1 ? switches / (length(x)-1) : missing,
            n_zero_to_one = count(==(1), d), n_one_to_zero = count(==(-1), d),
            longest_dwell_draws = maximum(diff(ends)), lag1_autocorrelation = lag1,
            constant_chain = switches == 0))
    end
    return DataFrame(rows)
end

function rank_convergence_diagnostics(wide::AbstractDataFrame)
    df = _diagnostics_summary_dataframe(dataframe_to_chains(wide))
    df.rhat_method = fill("rank-normalized split/folded (MCMCChains 7)", nrow(df))
    df.ess_bulk_method = fill("rank-normalized", nrow(df))
    df.ess_tail_method = fill("minimum ESS of 5% and 95% quantile indicators", nrow(df))
    df.constant_all_draws = [begin
        col = Symbol(string(param))
        col in propertynames(wide) && length(unique(wide[!, col])) == 1
    end for param in df.parameters]
    # Constant draws cannot demonstrate mixing, even if a library returns a
    # nominal ESS. Retain a flag and explicitly withhold convergence statistics.
    for col in (:rhat, :ess_bulk, :ess_tail)
        if col in propertynames(df)
            allowmissing!(df, col)
            df[df.constant_all_draws, col] .= missing
        end
    end
    return df
end

function _result_transitions(result::CoxSamplerResult)
    chains = result.samples isa MHSamples ? [result.samples] :
        result.samples isa MultiMHSamples ? result.samples.chains : MHSamples[]
    rows = NamedTuple[]
    for (c, samples) in enumerate(chains)
        isdefined(samples, :transitions) || continue
        for row in samples.transitions
            push!(rows, merge((chain = c,), row))
        end
    end
    return DataFrame(rows)
end

function transition_diagnostics_summary(transitions::AbstractDataFrame)
    isempty(transitions) && return DataFrame()
    rows = NamedTuple[]
    for g in groupby(transitions, [:chain, :phase])
        probs = collect(skipmissing(g.acceptance_probability))
        errs = [Float64(x) for x in skipmissing(g.energy_error) if isfinite(x)]
        push!(rows, (chain = first(g.chain), phase = first(g.phase), n_transitions = nrow(g),
            mean_acceptance_probability = isempty(probs) ? missing : mean(probs),
            n_divergent = count(identity, g.divergent), divergence_rate = mean(g.divergent),
            max_abs_finite_energy_error = isempty(errs) ? missing : maximum(abs, errs)))
    end
    return DataFrame(rows)
end

function save_revision_diagnostics(wide::AbstractDataFrame, output_dir::AbstractString;
    transitions = DataFrame(), include_rank::Bool = true)
    mkpath(output_dir)
    zero = zero_sum_diagnostics(wide)
    mixing = selection_mixing_diagnostics(wide)
    !isempty(zero) && _write_csv(joinpath(output_dir, "zero_sum_summary.csv"), zero)
    !isempty(mixing) && _write_csv(joinpath(output_dir, "selection_mixing.csv"), mixing)
    if include_rank
        _write_csv(joinpath(output_dir, "rank_diagnostics.csv"), rank_convergence_diagnostics(wide))
    end
    if !isempty(transitions)
        _write_csv(joinpath(output_dir, "transition_diagnostics.csv"), transitions)
        _write_csv(joinpath(output_dir, "transition_summary.csv"), transition_diagnostics_summary(transitions))
    end
    return (zero_sum = zero, selection_mixing = mixing)
end
