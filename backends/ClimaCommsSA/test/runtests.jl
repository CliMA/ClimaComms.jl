using Distributed
using Test

@assert nprocs() == 1
addprocs(3, exeflags = "--project=$(Base.active_project())")

@everywhere begin
    using Distributed
    using SharedArrays
    using ClimaComms
    using ClimaCommsSA
    include(joinpath(@__DIR__, "..", "..", "..", "test", "stencil.jl"))
end

@testset "Stencil" begin
    @everywhere stencil_test(ClimaCommsSA.SACommsContext, niterations = 100)
end
