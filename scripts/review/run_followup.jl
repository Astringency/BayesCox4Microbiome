#!/usr/bin/env julia
# Execute explicit supplementary simulation plans; the baseline grid uses run_simulation.sh.
using JSON, Dates

function followup_options(argv)
    length(argv) % 2 == 0 || error("Expected --key value pairs")
    opts = Dict(argv[i] => argv[i+1] for i in 1:2:length(argv))
    all(k in ("--plan", "--outdir", "--workers", "--dry-run") for k in keys(opts)) || error("Unknown option")
    return opts
end

function followup_command(project, out, job)
    method = job["method"]; car = method == "cox-rhs"
    tag = job["tag"]
    isabspath(tag) || ".." in splitpath(tag) ? error("Plan tag must stay inside output directory") : nothing
    options = [
        "outdir" => joinpath(out, "fits", tag), "reps" => max(50, maximum(job["reps"])),
        "rep-ids" => join(job["reps"], ','), "n" => job["n"], "p" => job["p"],
        "nonzero" => job["nonzero"], "baseline" => job["baseline"], "scenario" => job["scenario"],
        "priors" => method, "sampler" => get(job,"sampler","hmc"), "nuts-backend" => "explicit",
        "hmc-n-leapfrog" => job["leapfrog"], "target-accept" => job["target_accept"],
        "samples" => job["samples"], "warmup" => job["warmup"], "thin" => 1,
        "chains" => 4, "multichains" => true, "noncentered" => true, "init" => get(job,"init","rand"),
        "seed" => 1, "use-car" => car, "sample-car-strength" => car,
        "car-weight-quantile" => .99, "car-precision" => 1,
        "car-strength-prior-a" => 1, "car-strength-prior-b" => 1, "car-distance" => "bray",
        "p0" => "truth", "constraint" => job["constraint"], "prox-sum0-lambda" => job["sum_lambda"],
        "prox-l1-lambda" => get(job,"l1_lambda",1e-6),
        "alpha-prior-kind" => "invgamma", "alpha-prior-shape" => 1, "alpha-prior-scale" => job["p"],
        "sslasso-spike-scale" => .04, "sslasso-slab-scale" => .5,
        "sslasso-theta-a" => 1, "sslasso-theta-b" => 1,
        "rhs-slab-df" => 4, "rhs-slab-s2" => 2, "rhs-tau-scale" => 1, "rhs-local-scale" => 1,
        "pseudocount" => 0, "standardize" => false, "weak-multiplier" => .35,
        "diagnose" => false, "progress" => false, "resume" => true,
        "sample-thin" => 1, "first-rep-sample-thin" => 1,
        "save-wide-csv" => true, "save-jld2" => false]
    haskey(job,"max_depth") && push!(options,"max-depth" => job["max_depth"])
    argv = String[]
    for (k,v) in options; append!(argv,["--" * k,string(v)]); end
    return `$(Base.julia_cmd()) -t 4 --startup-file=no --project=$project $(joinpath(project,"scripts","run_simulation.jl")) $argv`
end

function followup_main(argv=ARGS)
    if isempty(argv) || "--help" in argv
        println("Usage: julia --project=. scripts/review/run_followup.jl --plan FILE --outdir DIR [--workers 1] [--dry-run true]")
        return
    end
    opts=followup_options(argv)
    project=normpath(joinpath(@__DIR__,"..",".."))
    plan=JSON.parsefile(opts["--plan"]); out=abspath(opts["--outdir"])
    jobs=plan["jobs"]; workers=parse(Int,get(opts,"--workers","1")); workers>0 || error("workers must be positive")
    tags=[j["tag"] for j in jobs]; allunique(tags) || error("Plan has overlapping output tags")
    commands=[followup_command(project,out,j) for j in jobs]
    if parse(Bool,get(opts,"--dry-run","false"))
        foreach(println,commands); return
    end
    mkpath(out)
    saved=joinpath(out,"plan.json")
    if isfile(saved)
        JSON.parsefile(saved)==plan || error("Existing plan differs; use a new output root")
    else
        write(saved,JSON.json(plan))
    end
    semaphore=Base.Semaphore(workers)
    codes=fill(-1,length(jobs))
    @sync for i in eachindex(jobs)
        @async Base.acquire(semaphore) do
            log=joinpath(out,"logs",tags[i]*".log"); mkpath(dirname(log))
            open(log,"a") do io
                proc=run(pipeline(ignorestatus(addenv(commands[i],"OPENBLAS_NUM_THREADS"=>"1","GKSwstype"=>"100")),stdout=io,stderr=io))
                codes[i]=proc.exitcode
            end
            println(tags[i]," exit=",codes[i]); flush(stdout)
        end
    end
    write(joinpath(out,"execution.json"),JSON.json(Dict("finished"=>string(now()),"tags"=>tags,"exit_codes"=>codes)))
    all(==(0),codes) || error("Some supplementary fits failed; inspect logs. No automatic retries.")
end

abspath(PROGRAM_FILE)==abspath(@__FILE__) && followup_main()
