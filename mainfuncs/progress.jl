const _PROGRESS_OUTPUT_LOCK = ReentrantLock()

function _progress_meter(
    enabled::Bool,
    total::Integer,
    default_desc::AbstractString;
    progress_desc::AbstractString = default_desc,
    progress_offset::Integer = 0,
)
    return enabled ? Progress(total; desc = String(progress_desc), showspeed = true, dt = 0.0, offset = Int(progress_offset)) : nothing
end

function _progress_next!(prog; kwargs...)
    prog === nothing && return nothing
    lock(_PROGRESS_OUTPUT_LOCK) do
        next!(prog; kwargs...)
    end
    return nothing
end

function _progress_finish!(prog)
    prog === nothing && return nothing
    lock(_PROGRESS_OUTPUT_LOCK) do
        finish!(prog)
    end
    return nothing
end

function _progress_hmc_next!(
    prog,
    stat::NamedTuple;
    progress_desc::AbstractString,
    progress_offset::Integer,
)
    _ = stat
    lock(_PROGRESS_OUTPUT_LOCK) do
        next!(prog; desc = progress_desc, offset = progress_offset, force = true)
    end
    return nothing
end

function _reserve_chain_progress_lines(chains::Integer, progress::Bool)
    if progress && chains > 1
        lock(_PROGRESS_OUTPUT_LOCK) do
            print(stderr, "\n" ^ (chains - 1))
            flush(stderr)
        end
    end
    return nothing
end

_chain_progress_offset(chain::Integer, chains::Integer) = chains - chain

function _chain_progress_desc(default_desc::AbstractString, chain::Integer, chains::Integer)
    return "$default_desc chain $chain/$chains"
end
