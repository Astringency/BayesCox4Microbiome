using BayesCox4Micro: _divergence_count
using Test
isdefined(Main, :fit_and_evaluate) || include(joinpath(@__DIR__, "..", "scripts", "lib", "analysis_common.jl"))

@testset "Tuning reads HMC transitions and excludes warmup" begin
    records = NamedTuple[(phase="warmup", divergent=true),
        (phase="sample", divergent=false), (phase="sample", divergent=true)]
    make_chain(transitions) = MHSamples(zeros(2,1), zeros(2), ["beta[1]"],
        0.9, 0.8, [0.01], transitions)
    chain = make_chain(records)
    @test _divergence_count((samples=chain,)) == 1
    @test _divergence_count((samples=MultiMHSamples([chain,chain]),)) == 2
    clean = make_chain(NamedTuple[(phase="warmup", divergent=true),
        (phase="sample", divergent=false)])
    @test _divergence_count((samples=clean,)) == 0
    absent = make_chain(NamedTuple[])
    @test ismissing(_divergence_count((samples=absent,)))
    @test ismissing(_divergence_count((samples=MultiMHSamples([chain,absent]),)))
    unknown = make_chain(NamedTuple[(phase="sample", divergent=missing)])
    @test ismissing(_divergence_count((samples=unknown,)))
end
