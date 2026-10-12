using SHA

# These helpers deliberately preserve the legacy RNG path for :baseline.
function resolve_simulation_scenario(value)
    s = lowercase(string(value))
    aliases = Dict("baseline" => :baseline, "s1" => :weak, "weak" => :weak,
        "s2" => :correlated_null, "correlated_null" => :correlated_null,
        "s3" => :counts, "counts" => :counts, "phylo" => :phylo,
        "s4" => :phylo_bm, "phylo_bm" => :phylo_bm)
    haskey(aliases, s) || throw(ArgumentError("scenario must be baseline, S1/weak, S2/correlated_null, S3/counts, S4/phylo_bm, or legacy phylo"))
    return aliases[s]
end

function simulation_scenario_parameters(args)
    return (scenario = resolve_simulation_scenario(arg(args, "scenario", "baseline")),
        weak_multiplier = arg(args, "weak-multiplier", 0.35),
        null_correlation = arg(args, "null-correlation", 0.6),
        depth_median = arg(args, "depth-median", 5000.0),
        depth_log_sd = arg(args, "depth-log-sd", 0.5))
end

const SIMULATION_SCENARIO_CLI_KEYS = ("scenario", "weak-multiplier", "null-correlation", "depth-median", "depth-log-sd")

function scenario_latent_parameters(p, idx, scenario; null_correlation = 0.6)
    mu = zeros(Float64, p)
    mu[idx] .= log(0.5p)
    covariance = paper_covariance(p, idx)
    paired_nulls = Int[]
    if scenario == :correlated_null
        length(idx) == 6 || throw(ArgumentError("S2 requires exactly six signal taxa"))
        0 <= null_correlation < 1 || throw(ArgumentError("null correlation must be in [0,1)"))
        nulls = setdiff(1:p, idx)
        length(nulls) >= 6 || throw(ArgumentError("S2 requires at least six null taxa"))
        paired_nulls = nulls[1:6]
        A = Matrix{Float64}(I, p, p)
        for (signal, null) in zip(idx, paired_nulls)
            A[null, null] = sqrt(1-null_correlation^2)
            A[null, signal] = null_correlation
        end
        # Constructive covariance guarantees positive definiteness and the
        # requested signal-null correlations, including induced null-null terms.
        covariance = Symmetric(A * covariance * A')
    elseif scenario == :counts
        # A fixed broad abundance profile produces nontrivial sampling zeros.
        # Signal taxa have moderate abundance; this is not the weak-signal case.
        mu .= collect(range(-7.5, 0.0; length = p))
        mu[idx] .= 0.0
    elseif scenario in (:phylo, :phylo_bm)
        covariance = scenario_phylogeny(p, idx, scenario).covariance
    end
    return (mu = mu, covariance = covariance, paired_nulls = paired_nulls)
end

function sequencing_observations(rng, latent_X; depth_median = 5000.0, depth_log_sd = 0.5)
    depth_median > 0 || throw(ArgumentError("depth median must be positive"))
    depth_log_sd >= 0 || throw(ArgumentError("depth log SD must be nonnegative"))
    n, p = size(latent_X)
    depths = max.(1, round.(Int, exp.(log(depth_median) .+ depth_log_sd .* randn(rng, n))))
    counts = Matrix{Int}(undef, n, p)
    for i in 1:n
        counts[i, :] .= rand(rng, Multinomial(depths[i], collect(view(latent_X, i, :))))
    end
    X = counts ./ depths
    return (X = X, sequencing_counts = counts, sequencing_depths = depths,
        observed_zero_fraction = mean(counts .== 0))
end

function scenario_metadata(sim; weak_multiplier = 0.35, null_correlation = 0.6,
    depth_median = 5000.0, depth_log_sd = 0.5, kwargs...)
    result = Dict{String,Any}("scenario" => string(sim.scenario))
    if sim.scenario == :weak
        result["weak_multiplier"] = weak_multiplier
    elseif sim.scenario == :correlated_null
        result["null_correlation"] = null_correlation
        result["correlated_null_indices"] = sim.paired_nulls
    elseif sim.scenario == :counts
        result["depth_median"] = depth_median
        result["depth_log_sd"] = depth_log_sd
        result["latent_mean_profile"] = "linear -7.5 to 0; signal means 0"
        result["measurement_model"] = "Multinomial(depth, latent softmax abundance)"
        result["zero_replacement"] = "half training minimum positive abundance, then closure"
    elseif sim.scenario == :phylo
        result["phylogenetic_covariance_range"] = 4.0
        result["phylogenetic_graph_range"] = 2.0
        result["phylogenetic_max_distance"] = 4
        result["phylogenetic_misspec_seed"] = 9017
    elseif sim.scenario == :phylo_bm
        phylo = scenario_phylogeny(length(sim.beta), sim.nonzero_index, sim.scenario)
        merge!(result, brownian_phylogeny_metadata(phylo))
    end
    return result
end

"""Planted balanced phylogeny, with signal taxa in neighboring leaves.

The graph is known from the generating tree, not estimated from observed
abundances. A fixed leaf-label permutation preserves the graph's spectrum,
degrees and edge weights while changing which taxa are neighbors.
"""
function planted_phylogeny(p::Integer, signal_indices; covariance_model::Symbol = :exponential)
    p >= 2 || throw(ArgumentError("phylogeny needs at least two taxa"))
    covariance_model in (:exponential, :brownian) || throw(ArgumentError("unsupported phylogenetic covariance"))
    order = vcat(collect(Int, signal_indices), setdiff(1:p, signal_indices))
    length(order) == p && length(unique(order)) == p || throw(ArgumentError("invalid signal indices"))
    tree_size = nextpow(2, p)
    D = zeros(Float64, p, p)
    for i in 1:p, j in (i+1):p
        left, right, distance = tree_size + i - 1, tree_size + j - 1, 0
        while left != right
            left, right = left ÷ 2, right ÷ 2
            distance += 2
        end
        D[order[i], order[j]] = D[order[j], order[i]] = distance
    end
    # Edge IDs are the child IDs of the implicit binary heap. Keep only edges
    # ancestral to retained tips; unit lengths preserve an ultrametric tree.
    height = trailing_zeros(tree_size)
    paths = [Int[] for _ in 1:p]
    for i in 1:p
        node = tree_size + i - 1
        while node > 1
            push!(paths[order[i]], node)
            node ÷= 2
        end
    end
    children = sort(unique(vcat(paths...)))
    incidence = Float64[child in paths[i] for i in 1:p, child in children]
    branch_lengths = ones(length(children))
    shared_path = incidence * Diagonal(branch_lengths) * incidence'
    covariance = covariance_model == :brownian ? Symmetric(shared_path / height) : Symmetric(exp.(-D / 4.0))
    W = [i != j && D[i,j] <= 4 ? exp(-D[i,j]/2.0) : 0.0 for i in 1:p, j in 1:p]
    permutation = randperm(MersenneTwister(9017), p)
    W_wrong = W[permutation, permutation]
    # Collapse unary nodes when exporting Newick, preserving total branch length.
    active = Set(children)
    function subtree(node)
        node >= tree_size && return ("x$(order[node-tree_size+1])", 1)
        descendants = [subtree(child) for child in (2node, 2node+1) if child in active]
        length(descendants) == 1 && return (descendants[1][1], descendants[1][2]+1)
        return ("(" * join(["$(s):$(ell)" for (s, ell) in descendants], ",") * ")", 1)
    end
    newick = subtree(1)[1] * ";"
    return (leaf_order = order, distance = D, covariance = covariance, weights = W,
        misspecified_weights = W_wrong, permutation = permutation,
        covariance_model = covariance_model, height = height, edge_parent = children .÷ 2,
        edge_child = children, branch_lengths = branch_lengths, incidence = incidence,
        shared_path = shared_path, newick = newick)
end

scenario_phylogeny(p, indices, scenario) = planted_phylogeny(p, indices;
    covariance_model = resolve_simulation_scenario(scenario) == :phylo_bm ? :brownian : :exponential)

const BROWNIAN_DGP_METADATA_KEYS = ("dgp_version", "phylogenetic_covariance_model",
    "phylogenetic_root_state", "phylogenetic_tree_height", "phylogenetic_diffusion_rate",
    "phylogenetic_tree_sha256", "phylogenetic_covariance_sha256", "phylogenetic_graph_sha256",
    "phylogenetic_graph_range", "phylogenetic_max_distance", "phylogenetic_misspec_seed")

function brownian_phylogeny_metadata(phylo)
    phylo.covariance_model == :brownian || error("expected Brownian phylogeny")
    digest(x) = bytes2hex(sha256(join(string.(vec(x)), ",")))
    return Dict{String,Any}("dgp_version" => "phylo_bm_v1", "phylogenetic_covariance_model" => "brownian_shared_path",
        "phylogenetic_root_state" => 0.0, "phylogenetic_tree_height" => phylo.height,
        "phylogenetic_diffusion_rate" => 1.0 / phylo.height,
        "phylogenetic_tree_sha256" => bytes2hex(sha256(phylo.newick)),
        "phylogenetic_covariance_sha256" => digest(phylo.covariance),
        "phylogenetic_graph_sha256" => digest(vcat(vec(phylo.weights), vec(phylo.misspecified_weights))),
        "phylogenetic_graph_range" => 2.0, "phylogenetic_max_distance" => 4,
        "phylogenetic_misspec_seed" => 9017)
end

function save_planted_phylogeny(directory, phylo)
    mkpath(directory)
    if phylo.covariance_model == :brownian
        manifest = brownian_phylogeny_metadata(phylo)
        path = joinpath(directory, "phylogeny.json")
        if isfile(path)
            JSON.parsefile(path) == manifest || error("incompatible saved phylogeny; use a new output directory")
            if all(isfile(joinpath(directory, name)) for name in ("tree.newick", "edges.csv",
                "patristic_distance.csv", "correct_weights.csv", "misspecified_weights.csv",
                "latent_covariance.csv", "leaf_order.csv", "misspecification_permutation.csv"))
                return nothing
            end
        elseif isfile(joinpath(directory, "latent_covariance.csv"))
            error("unversioned phylogeny already exists; use a separate S4 output directory")
        end
    end
    for (name, matrix) in (("patristic_distance", phylo.distance), ("correct_weights", phylo.weights),
        ("misspecified_weights", phylo.misspecified_weights), ("latent_covariance", phylo.covariance))
        CSV.write(joinpath(directory, name * ".csv"), DataFrame(matrix, Symbol.("x" .* string.(1:size(matrix,2)))))
    end
    CSV.write(joinpath(directory, "leaf_order.csv"), DataFrame(leaf_position = 1:length(phylo.leaf_order), taxon_index = phylo.leaf_order))
    CSV.write(joinpath(directory, "misspecification_permutation.csv"), DataFrame(taxon_index = 1:length(phylo.permutation), reassigned_index = phylo.permutation))
    CSV.write(joinpath(directory, "edges.csv"), DataFrame(parent = phylo.edge_parent, child = phylo.edge_child,
        branch_length = phylo.branch_lengths))
    write(joinpath(directory, "tree.newick"), phylo.newick * "\n")
    if phylo.covariance_model == :brownian
        # Publish only after all tree artifacts are written. Formal partitions
        # use separate output roots, avoiding concurrent writers altogether.
        open(joinpath(directory, "phylogeny.json"), "w") do io
            JSON.print(io, brownian_phylogeny_metadata(phylo), 2)
        end
    end
    return nothing
end

function save_scenario_observations(sim, directory, label)
    mkpath(directory)
    observed = DataFrame(sim.X, Symbol.("x" .* string.(1:size(sim.X, 2))))
    observed.time, observed.event = sim.time, sim.event
    observed.event_time, observed.censor_time = sim.event_time, sim.censor_time
    if sim.sequencing_depths !== nothing
        observed.sequencing_depth = sim.sequencing_depths
        CSV.write(joinpath(directory, label * "_counts.csv"), DataFrame(sim.sequencing_counts, :auto))
        CSV.write(joinpath(directory, label * "_latent.csv"), DataFrame(sim.latent_X, :auto))
    end
    CSV.write(joinpath(directory, label * ".csv"), observed)
    metadata = Dict("scenario" => string(sim.scenario), "beta" => sim.beta,
        "paired_null_indices" => sim.paired_nulls, "observed_zero_fraction" => sim.observed_zero_fraction,
        "empirical_depth_median" => sim.sequencing_depths === nothing ? nothing : median(sim.sequencing_depths),
        "latent_covariance" => [collect(row) for row in eachrow(sim.latent_covariance)])
    sim.scenario == :phylo_bm && merge!(metadata, scenario_metadata(sim))
    open(joinpath(directory, label * ".json"), "w") do io
        JSON.print(io, metadata, 2)
    end
    return nothing
end
