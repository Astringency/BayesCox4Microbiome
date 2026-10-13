#!/usr/bin/env julia
# Rebuild training cohorts from the frozen current generator; copy actual saved test data.
include(joinpath(ARGS[1], "scripts", "analysis_common.jl"))
using SHA
inventory=CSV.read(ARGS[2],DataFrame)
out=ARGS[3];mkpath(out)
records=Dict{String,Any}[]
seen=Set()
for row in eachrow(inventory)
    row.method=="official-bccoxph" && continue
    meta=JSON.parsefile(joinpath(row.source,"metadata.json"))
    get(meta,"scenario","baseline")=="baseline" || continue
    key=(meta["n"],meta["p"],meta["nonzero_count"],meta["baseline"],meta["seed"])
    key in seen && continue
    push!(seen,key)
    n,p,nz,baseline,seed=key
    label="rep$(seed)_$(baseline)_n$(n)_p$(p)_nz$(nz)_seed$(seed)"
    spec=JSON.parsefile(replace(meta["test_data"],r"\.csv$"=>".json"))
    sim=generate_simulation_data(n,p,nz;baseline=Symbol(baseline),seed,
        baseline_rate=spec["baseline_rate"],baseline_shape=spec["baseline_shape"])
    sim.censoring_upper==spec["censoring_upper"] || error("censoring specification mismatch")
    sim.beta==spec["beta"] || error("coefficient mismatch")
    train=DataFrame(sim.X,Symbol.("x" .* string.(1:p)))
    train.time=sim.time;train.event=sim.event
    CSV.write(joinpath(out,label*"_train.csv"),train)
    cp(meta["test_data"],joinpath(out,label*"_test.csv");force=true)
    CSV.write(joinpath(out,label*"_beta.csv"),DataFrame(variable="x" .* string.(1:p),beta=sim.beta))
    push!(records,Dict("label"=>label,"source"=>row.source,"saved_test"=>meta["test_data"],
        "saved_test_sha256"=>bytes2hex(sha256(read(meta["test_data"]))),
        "training_source"=>"regenerated from frozen generator and recorded DGP/seed; original baseline training array was not saved"))
end
open(joinpath(out,"manifest.json"),"w") do io;JSON.print(io,records,2);end
println("Exported $(length(records)) paired cohorts")
