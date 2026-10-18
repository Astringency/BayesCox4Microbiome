# Fit-level wall-clock partition. Parallel chains are not synchronized: the
# warmup window ends when the last chain finishes warmup, so it can overlap
# retained draws from faster chains. For serial chains, sum the disjoint warmup
# windows. The remaining timed work (including draw conversion) is sampling.
mutable struct _SamplingPhaseClock
    start_ns::UInt64
    warmup_end_ns::Vector{UInt64}
    chain_end_ns::Vector{UInt64}
end

_SamplingPhaseClock(chains::Integer) =
    _SamplingPhaseClock(0, zeros(UInt64, chains), zeros(UInt64, chains))

_start_sampling_clock!(clock::_SamplingPhaseClock) = (clock.start_ns = time_ns())
_finish_warmup!(::Nothing, ::Integer) = nothing
_finish_warmup!(clock::_SamplingPhaseClock, chain::Integer) =
    (clock.warmup_end_ns[chain] = time_ns())
_finish_sampling_chain!(::Nothing, ::Integer) = nothing
_finish_sampling_chain!(clock::_SamplingPhaseClock, chain::Integer) =
    (clock.chain_end_ns[chain] = time_ns())

function _phase_diagnostics(timed, clock::_SamplingPhaseClock, warmup::Integer,
    parallel::Bool)
    warmup_seconds = if warmup == 0
        0.0
    elseif any(iszero, clock.warmup_end_ns) || any(iszero, clock.chain_end_ns)
        missing
    elseif parallel
        Float64(maximum(clock.warmup_end_ns) - clock.start_ns) / 1e9
    else
        elapsed_ns = UInt64(0)
        start_ns = clock.start_ns
        for c in eachindex(clock.warmup_end_ns)
            elapsed_ns += clock.warmup_end_ns[c] - start_ns
            start_ns = clock.chain_end_ns[c]
        end
        Float64(elapsed_ns) / 1e9
    end
    if !ismissing(warmup_seconds)
        warmup_seconds = clamp(warmup_seconds, 0.0, timed.time)
    end
    sample_seconds = ismissing(warmup_seconds) ? missing : timed.time - warmup_seconds
    gc_percent = timed.time > 0 ? 100 * timed.gctime / timed.time : NaN
    return SamplingDiagnostics(warmup_seconds, sample_seconds, timed.time,
        Int64(timed.bytes), timed.gctime, gc_percent)
end

# AdvancedHMC's progress callback is disabled with progress=false. Delegate its
# complete adaptation step instead, preserving the original transition/RNG
# sequence and recording the boundary after the final warmup adaptation.
struct _WarmupTimingAdaptor{A} <: AdvancedHMC.Adaptation.AbstractAdaptor
    inner::A
    clock::_SamplingPhaseClock
    chain::Int
end

_timing_adaptor(adaptor, ::Nothing, ::Integer) = adaptor
_timing_adaptor(adaptor, clock::_SamplingPhaseClock, chain::Integer) =
    _WarmupTimingAdaptor(adaptor, clock, Int(chain))

function AdvancedHMC.Adaptation.adapt!(
    h::AdvancedHMC.Hamiltonian,
    kernel::AdvancedHMC.AbstractMCMCKernel,
    adaptor::_WarmupTimingAdaptor,
    iteration::Int,
    n_adapts::Int,
    position::AdvancedHMC.AbstractVecOrMat{<:AbstractFloat},
    acceptance::AdvancedHMC.AbstractScalarOrVec{<:AbstractFloat},
)
    result = AdvancedHMC.Adaptation.adapt!(h, kernel, adaptor.inner,
        iteration, n_adapts, position, acceptance)
    iteration == n_adapts && _finish_warmup!(adaptor.clock, adaptor.chain)
    return result
end
