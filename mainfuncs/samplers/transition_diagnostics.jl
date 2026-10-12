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
