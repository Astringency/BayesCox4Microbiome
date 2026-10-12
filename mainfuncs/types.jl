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
    tie_method::Symbol
    transform::Symbol
    pseudocount::Float64
    closure::Bool
    standardize::Bool
    col_mean::Vector{Float64}
    col_scale::Vector{Float64}
end

struct CARWeightEdges
    i::Vector{Int}
    j::Vector{Int}
    w::Vector{Float64}
    p::Int
end

struct MHSamples
    draws::Matrix{Float64}
    logposterior::Vector{Float64}
    names::Vector{String}
    acceptance_rate::Float64
    warmup_acceptance_rate::Float64
    final_step::Vector{Float64}
    transitions::Vector{NamedTuple}
end

# Backward-compatible constructor for samplers without HMC transition records.
MHSamples(draws, logposterior, names, acceptance_rate, warmup_acceptance_rate, final_step) =
    MHSamples(draws, logposterior, names, acceptance_rate, warmup_acceptance_rate, final_step, NamedTuple[])

struct MultiMHSamples
    chains::Vector{MHSamples}
end

struct SamplingDiagnostics
    warmup_seconds::Union{Missing,Float64}
    sample_seconds::Union{Missing,Float64}
    total_seconds::Float64
    allocated_bytes::Int64
    gc_seconds::Float64
    gc_percent::Float64
end

SamplingDiagnostics() = SamplingDiagnostics(missing, missing, NaN, 0, 0.0, NaN)

struct CoxSamplerResult{S,P}
    samples::S
    data::CoxPreparedData
    prior::P
    sampler::Symbol
    diagnostics::SamplingDiagnostics
end

CoxSamplerResult(samples, data, prior, sampler) =
    CoxSamplerResult(samples, data, prior, sampler, SamplingDiagnostics())
