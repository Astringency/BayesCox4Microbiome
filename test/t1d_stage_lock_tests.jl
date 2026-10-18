# Linux concurrency test only; no posterior is sampled.
using Test
include(isempty(ARGS) ? joinpath(@__DIR__, "..", "scripts", "lib", "realdata", "t1d.jl") : ARGS[1])

@testset "T1D stage ownership lock" begin
    @test Sys.islinux()
    @test Threads.nthreads() >= 2
    source = mktempdir(; cleanup=false)
    ready = Threads.Atomic{Int}(0)
    events = Tuple{Int,Symbol}[]
    event_lock = ReentrantLock()
    Threads.@threads for id in 1:2
        Threads.atomic_add!(ready, 1)
        while ready[] < 2; yield(); end
        with_t1d_stage_lock(source, "1") do
            lock(event_lock) do; push!(events, (id, :start)); end
            ccall(:usleep, Cint, (Cuint,), 250_000)
            lock(event_lock) do; push!(events, (id, :end)); end
        end
    end
    @test length(events) == 4
    @test events[1][1] == events[2][1]
    @test events[3][1] == events[4][1]
    @test events[1][1] != events[3][1]
    @test last.(events) == [:start, :end, :start, :end]
    @test isfile(joinpath(source, "stage_locks", "1.lock"))
    @test_throws ErrorException with_t1d_stage_lock(() -> nothing, source, "../invalid")
    @test with_t1d_stage_lock(() -> :reacquired, source, "1") == :reacquired
    println("Retained stage-lock test directory: ", source)
end
