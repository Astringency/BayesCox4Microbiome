# Shared candidate grids for real-data and simulation tuning.
function _grid_values(value)
    value isa AbstractString && return Any[value]
    value isa AbstractVector && return collect(Any, value)
    value isa Tuple && return collect(Any, value)
    return Any[value]
end

function _expand_grid_values(grid::Vector{Pair{K,Any}}) where {K}
    combos = [Dict{K,Any}()]
    for (key, values) in grid
        next_combos = Dict{K,Any}[]
        for combo in combos, value in _grid_values(values)
            expanded = copy(combo)
            expanded[key] = value
            push!(next_combos, expanded)
        end
        combos = next_combos
    end
    return combos
end

function _expand_named_grid_values(grids::Dict{Name,Vector{Pair{K,Any}}}) where {Name,K}
    out = Dict{Name,Vector{Dict{K,Any}}}()

    for (name, grid) in pairs(grids)
        combos = _expand_grid_values(grid)

        combos = filter(combos) do cfg
            has_spike = haskey(cfg, K("sslasso_spike_scale"))
            has_slab  = haskey(cfg, K("sslasso_slab_scale"))

            if has_spike && has_slab
                cfg[K("sslasso_spike_scale")] <
                cfg[K("sslasso_slab_scale")]
            else
                true
            end
        end

        out[name] = combos
    end

    return out
end

# Edit these candidate-value lists directly to change the tuning search space.
# The script forms the Cartesian product within each prior/sampler grid.
const PRIOR_PARAM_GRID_VALUES = Dict{String,Vector{Pair{String,Any}}}(
    "hs" => Pair{String,Any}[
        "p0" => Any[nothing],
        "hs_local_scale" => [0.5, 1.0],
        "hs_tau_scale" => [1.0],
    ],
    "rhs" => Pair{String,Any}[
        "p0" => Any[nothing],
        "rhs_local_scale" => [0.5, 1.0],
        "rhs_tau_scale" => [1.0],
        "rhs_slab_df" => [4.0],
        "rhs_slab_s2" => [2.0],
    ],
    "new-rhs" => Pair{String,Any}[
        "p0" => Any[nothing],
        "rhs_local_scale" => [0.5, 1.0],
        "rhs_slab_df" => [4.0],
        "rhs_slab_s2" => [2.0],
    ],
    "est-rhs" => Pair{String,Any}[
        "p0" => Any[nothing],
        "rhs_local_scale" => [0.5, 1.0],
        "rhs_slab_df" => [4.0],
        "rhs_slab_s2" => [2.0],
    ],
    "cox-hs" => Pair{String,Any}[
        "p0" => Any[nothing],
        "hs_local_scale" => [0.5, 1.0],
    ],
    "cox-rhs" => Pair{String,Any}[
        "p0" => Any[nothing],
        "rhs_local_scale" => [0.5, 1.0],
        "rhs_slab_df" => [4.0],
        "rhs_slab_s2" => [2.0],
    ],
    "lasso" => Pair{String,Any}[
        "lasso_lambda" => Any[nothing],
        "lasso_lambda2_shape" => [1.0],
        "lasso_lambda2_rate" => [0.1, 0.5, 1.0, 2.0, 10.0],
    ],
    "sslasso" => Pair{String,Any}[
        "sslasso_spike_scale" =>  [0.04, 0.1, 0.5, 1.0, 2.0],
        "sslasso_slab_scale" => [0.5, 1.0, 2.0, 5.0, 10.0],
        "sslasso_theta_a" => [1.0],
        "sslasso_theta_b" => [1.0],
    ],
    "prox-lasso" => Pair{String,Any}[
        "prox_l1_lambda" => [1e-5, 1e-4, 1e-3],
        "alpha_prior_shape" => [1.0],
    ],
    "prox-sslasso" => Pair{String,Any}[
        "sslasso_spike_scale" => [0.04, 0.1, 0.5, 1.0, 2.0],
        "sslasso_slab_scale" => [0.5, 1.0, 2.0, 5.0, 10.0],
        "prox_l1_lambda" => [1e-5, 1e-4, 1e-3],
        "alpha_prior_shape" => [1.0],
    ],
)
        # "alpha_prior_shape" => [0.1, 0.5, 1.0],
        # "alpha_prior_scale" => [1.0, 5.0, 10.0],

const PRIOR_PARAM_GRIDS = _expand_named_grid_values(PRIOR_PARAM_GRID_VALUES)

const ALGORITHM_PARAM_GRID_VALUES = Dict{Symbol,Vector{Pair{Symbol,Any}}}(
    :nuts => Pair{Symbol,Any}[
        :sampler => [:nuts],
        :nuts_backend => [:turing, :explicit],
        :target_accept => [0.8, 0.9],
        :max_depth => [10],
    ],
    :nuts_turing => Pair{Symbol,Any}[
        :sampler => [:nuts_turing],
        :nuts_backend => [:turing],
        :target_accept => [0.8, 0.9],
        :max_depth => [10],
    ],
    :nuts_explicit => Pair{Symbol,Any}[
        :sampler => [:nuts_explicit],
        :nuts_backend => [:explicit],
        :target_accept => [0.8, 0.9],
        :max_depth => [10],
    ],
    :hmc => Pair{Symbol,Any}[
        :sampler => [:hmc],
        :hmc_n_leapfrog => [10, 20],
        :hmc_step_size => Any[nothing],
    ],
    :mh => Pair{Symbol,Any}[
        :sampler => [:mh],
    ],
    :gibbs => Pair{Symbol,Any}[
        :sampler => [:gibbs],
    ],
)

const ALGORITHM_PARAM_GRIDS = _expand_named_grid_values(ALGORITHM_PARAM_GRID_VALUES)

