using Distributed
using Test

@testset "Stencil" begin
    if nprocs() < 4
        addprocs(4 - nprocs(), exeflags="--project=$(Base.active_project())")
    end
    @everywhere using Distributed, SharedArrays, Pkg
    @everywhere Pkg.activate($(Base.active_project()))
    @everywhere include(joinpath(@__DIR__, "sa_stencil.jl"))
end
