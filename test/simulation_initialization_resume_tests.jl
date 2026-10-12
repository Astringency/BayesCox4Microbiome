using Test
if !isdefined(Main, :simulation_metadata_matches_typed)
    include(joinpath(@__DIR__, "..", "scripts", "analysis_common.jl"))
end

@testset "Resume requires matching recorded initialization" begin
    mktempdir() do dir
        expected = Dict("prior"=>"prox-sslasso", "sampler"=>"nuts", "init"=>"rand")
        function save(meta)
            open(joinpath(dir, "metadata.json"), "w") do io
                JSON.print(io, meta)
            end
        end
        save(expected)
        @test simulation_metadata_matches_typed(dir, expected)
        @test !simulation_metadata_matches_typed(dir, merge(expected, Dict("init"=>"zero")))
        save(Dict("prior"=>"prox-sslasso", "sampler"=>"nuts"))
        @test !simulation_metadata_matches_typed(dir, expected)
        @test !simulation_metadata_matches_typed(dir, merge(expected, Dict("init"=>"zero")))
    end
end
