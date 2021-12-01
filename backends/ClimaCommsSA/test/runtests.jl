using Distributed
using Test

function rundistributed(file; ntasks = 1)
    if nprocs() < ntasks
        addprocs(ntasks - nprocs())
    end
    @everywhere using Distributed, SharedArrays, Pkg
    @everywhere Pkg.activate($(Base.active_project()))
    @everywhere include(file)
end

@testset "Stencil" begin
    rundistributed(joinpath(@__DIR__, "sa_stencil.jl"), ntasks = 4)
end
