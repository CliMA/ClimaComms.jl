using CUDA, CUDA_Runtime_jll
using Test

include("basic.jl")

using MPI

function runmpi(file; ntasks = 1)
    # See https://github.com/JuliaParallel/MPI.jl/issues/820
    mpiexec = get(ENV, "MPITRAMPOLINE_MPIEXEC", MPI.mpiexec())
    Base.run(
        `$mpiexec -n $ntasks $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $file`,
    )
end

@test success(runmpi(joinpath(@__DIR__, "basic.jl"), ntasks = 2))
