# Chain-dispatch ownership probe only: the sampler is a stub, so no posterior is sampled.
using BayesCox4Micro, Random, LinearAlgebra, JSON
const B=BayesCox4Micro
BLAS.set_num_threads(1)
@eval B begin
    const legacy_target_capture=Any[]
    const legacy_capture_lock=ReentrantLock()
    function _legacy_capture_hmc_one(target;kwargs...)
        lock(legacy_capture_lock) do
            push!(legacy_target_capture,target)
        end
        return MHSamples(zeros(1,1), zeros(1), ["probe"], 1.0, NaN, [NaN])
    end
end
commits=["a5f9ff573f8ef88bcd95f5d31ed7d10b6b3c847a","59bb5dedd6d8b09bad51349735013f4bbe2e7f38",
    "1402e3a9e381de8d00f1625e4fe3573f2259d3a4","fcaa4b95ca08dc144a239f1f20069a6d8f168edd"]
results=Any[]
for (i,commit) in enumerate(commits)
    source=read(`git show $(commit * ":mainfuncs/samplers/hmc_sampler.jl")`,String)
    start=first(findfirst("function _sample_hmc_explicit(",source))
    body=source[start:end]
    name="_legacy_probe_"*string(i)
    body=replace(body,"_sample_hmc_explicit("=>name*"(","_sample_explicit_hmc_one("=>"_legacy_capture_hmc_one(")
    Base.include_string(B,body,"legacy_rhs_probe_"*commit)
    f=getfield(B,Symbol(name))
    boxes=0
    for n in names(B;all=true)
        startswith(string(n),"#"*name*"#") || continue
        for m in methods(getfield(B,n))
            lowered=Base.uncompressed_ast(m)
            boxes+=count(stmt -> stmt isa Expr && stmt.head == :(=) && stmt.args[1] isa Core.SlotNumber &&
                occursin("Core.Box",string(stmt)) && lowered.slotnames[stmt.args[1].id]==:target,lowered.code)
        end
    end
    trials=Any[]
    for trial in 1:(isempty(ARGS) ? 1000 : parse(Int, first(ARGS)))
        rng=MersenneTwister(925)
        X=exp.(randn(rng,80,12));t=collect(1.:80.);e=ones(Int,80)
        data=prepare_cox_data(X,t,e)
        empty!(B.legacy_target_capture)
        Base.invokelatest(f,data,RegularizedHorseshoePrior(p0=4.0);n_samples=12,warmup=12,chains=4,
            hmc_n_leapfrog=5,noncentered=true,constraint=:prox,multichains=true,rng=MersenneTwister(trial),progress=false)
        ids=objectid.(B.legacy_target_capture)
        push!(trials,Dict("trial"=>trial,"n_target_references"=>length(ids),"n_unique_targets"=>length(unique(ids))))
    end
    push!(results,Dict("commit"=>commit,"boxed_target_bindings"=>boxes,"trials"=>trials))
    println(commit," boxes=",boxes," trials=",length(trials)," shared target trials=",count(x -> x["n_unique_targets"] < 4,trials));flush(stdout)
end
open((length(ARGS) >= 2 ? ARGS[2] : "legacy_rhs_target_stress.json"),"w") do io;JSON.print(io,results,2);end


