struct CoxRiskSet
    order::Vector{Int}
    group_start::Vector{Int}
    group_stop::Vector{Int}
    obs_to_group::Vector{Int}
end

struct CoxPreparedData
    z::Matrix{Float64}
    time::Vector{Float64}
    event::Vector{Int}
    risk::CoxRiskSet
    transform::Symbol
    pseudocount::Float64
    closure::Bool
    standardize::Bool
    col_mean::Vector{Float64}
    col_scale::Vector{Float64}
end

struct MHSamples
    draws::Matrix{Float64}
    logposterior::Vector{Float64}
    names::Vector{String}
    acceptance_rate::Float64
    warmup_acceptance_rate::Float64
    final_step::Vector{Float64}
end

struct CoxSamplerResult{S,P}
    samples::S
    data::CoxPreparedData
    prior::P
    sampler::Symbol
end
