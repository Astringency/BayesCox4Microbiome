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
        finite_values(col) = col in propertynames(g) ? Float64[x for x in skipmissing(g[!, col]) if isfinite(x)] : Float64[]
        count_flags(col) = col in propertynames(g) && !any(ismissing, g[!, col]) ? count(!iszero, g[!, col]) : missing
        probs = finite_values(:acceptance_probability)
        steps = finite_values(:n_leapfrog)
        depths = finite_values(:tree_depth)
        sizes = finite_values(:step_size)
        errs = [Float64(x) for x in skipmissing(g.energy_error) if isfinite(x)]
        push!(rows, (chain = first(g.chain), phase = first(g.phase), n_transitions = nrow(g),
            mean_acceptance_probability = isempty(probs) ? missing : mean(probs),
            n_divergent = count_flags(:divergent), divergence_rate = count_flags(:divergent) / nrow(g),
            n_numerical_errors = count_flags(:numerical_error),
            n_nonfinite_energy_errors = count(x -> !isfinite(x), skipmissing(g.energy_error)),
            total_leapfrog_steps = isempty(steps) ? missing : sum(steps),
            max_tree_depth = isempty(depths) ? missing : maximum(depths),
            mean_step_size = isempty(sizes) ? missing : mean(sizes),
            max_abs_finite_energy_error = isempty(errs) ? missing : maximum(abs, errs)))
    end
    return DataFrame(rows)
end


function _transition_value(stat, keys, default = missing)
    for key in keys
        hasproperty(stat, key) && return getproperty(stat, key)
    end
    return default
end

function _transition_record(stat, iteration::Integer, warmup::Integer; divergence_threshold = 1000.0)
    energy_error = _transition_value(stat, (:hamiltonian_energy_error, :energy_error))
    numerical_error = _transition_value(stat, (:numerical_error,), false)
    divergent = _transition_value(stat, (:divergent,), missing)
    if ismissing(divergent)
        divergent = numerical_error || (!ismissing(energy_error) &&
            (!isfinite(energy_error) || abs(energy_error) > divergence_threshold))
    end
    return (iteration = Int(iteration), phase = iteration <= warmup ? "warmup" : "sample",
        acceptance_probability = _transition_value(stat, (:acceptance_rate, :accept_prob)),
        accepted = _transition_value(stat, (:is_accept, :accepted)),
        divergent = divergent, numerical_error = numerical_error, energy_error = energy_error,
        tree_depth = _transition_value(stat, (:tree_depth, :depth)),
        n_leapfrog = _transition_value(stat, (:n_steps, :n_leapfrog)),
        step_size = _transition_value(stat, (:step_size, :nom_step_size)))
end

_transition_records(stats, warmup::Integer) =
    NamedTuple[_transition_record(stat, i, warmup) for (i, stat) in enumerate(stats)]

function _fit_diagnostics(result)
    diag = sampler_diagnostics(result)
    accept = _sampler_acceptance(result)
    chain_diag = _chain_diagnostics(result)
    divergences = _divergence_count(result)
    rhat_max = get(chain_diag, :rhat_max, missing)
    ess_min = get(chain_diag, :ess_bulk_min, missing)
    acc = get(accept, :acceptance_rate, missing)
    rhat_ok = rhat_max === missing || Float64(rhat_max) <= 1.10
    ess_ok = ess_min === missing || Float64(ess_min) >= 10.0
    div_ok = divergences === missing || Int(divergences) == 0
    acc_ok = acc === missing || 0.05 <= Float64(acc) <= 1.0
    out = Dict{Symbol,Any}(
        :warmup_seconds => diag.warmup_seconds,
        :sample_seconds => diag.sample_seconds,
        :total_seconds => diag.total_seconds,
        :allocated_mb => diag.allocated_bytes / 2.0^20,
        :gc_seconds => diag.gc_seconds,
        :gc_percent => diag.gc_percent,
        :divergences => divergences,
        :converged => rhat_ok && ess_ok && div_ok && acc_ok,
    )
    merge!(out, accept)
    merge!(out, chain_diag)
    return out
end

function _sampler_acceptance(result)
    if result.samples isa MHSamples
        return Dict{Symbol,Any}(
            :acceptance_rate => result.samples.acceptance_rate,
            :warmup_acceptance_rate => result.samples.warmup_acceptance_rate,
            :final_step_mean => mean(result.samples.final_step),
        )
    elseif result.samples isa MultiMHSamples
        samples = result.samples.chains
        return Dict{Symbol,Any}(
            :acceptance_rate => mean([s.acceptance_rate for s in samples]),
            :warmup_acceptance_rate => mean([s.warmup_acceptance_rate for s in samples]),
            :final_step_mean => mean([mean(s.final_step) for s in samples]),
        )
    end
    return Dict{Symbol,Any}(
        :acceptance_rate => missing,
        :warmup_acceptance_rate => missing,
        :final_step_mean => missing,
    )
end

function _divergence_count(result)
    if result.samples isa Union{MHSamples,MultiMHSamples}
        chains = result.samples isa MHSamples ? [result.samples] : result.samples.chains
        total = 0
        for chain in chains
            retained = [r for r in chain.transitions if r.phase == "sample"]
            isempty(retained) && return missing
            any(r -> ismissing(r.divergent), retained) && return missing
            total += count(r -> r.divergent, retained)
        end
        return total
    end
    wide = try
        samples_wide_dataframe(result)
    catch
        return missing
    end
    for col in (:divergent, :is_divergent, :numerical_error)
        if col in propertynames(wide)
            return count(!=(0), Int.(wide[!, col]))
        end
    end
    return missing
end

