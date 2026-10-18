module ScaleCalibrationTests
using Test, BayesCox4Micro, Random, LinearAlgebra, Statistics
include(joinpath(@__DIR__,"..","scripts","lib","simulation","scale_calibration.jl"))
@testset "Training null-information scale calibration, including Efron ties" begin
    rng=MersenneTwister(19)
    X=rand(rng,30,5).+.05;X./=sum(X;dims=2)
    time=Float64.(repeat(1:15,inner=2));event=Int.(mod.(1:30,3).!=0)
    info=null_cox_information(X,time,event)
    data=prepare_cox_data(X,time,event)
    h=1e-4
    numeric=zeros(5)
    for j in 1:5
        b=zeros(5);b[j]=h
        numeric[j]=-(cox_logpartiallikelihood(data,b)-2cox_logpartiallikelihood(data,zeros(5))+
            cox_logpartiallikelihood(data,-b))/h^2
    end
    @test info≈numeric rtol=1e-5
    cal=calibrate_rhs_scale(X,time,event;p0=2.)
    @test cal["expected_signal_count"]≈2. atol=1e-12
    @test cal["tau_default"]==cox_tau0(5,2.,count(!iszero,event))
    @test cal["tau_information"]≈cal["tau_default"]*cal["multiplier"]
    @test_throws ArgumentError calibrate_rhs_scale(X,time,zeros(Int,30);p0=2.)
    @test_throws ArgumentError calibrate_rhs_scale(X,time,event;p0=5.)
end
end
