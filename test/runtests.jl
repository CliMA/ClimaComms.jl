using CUDA, CUDA_Runtime_jll

include("basic.jl")

using MPI, Test

function runmpi(file; ntasks = 1)
    Base.run(
        `$(MPI.mpiexec()) -n $ntasks $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $file`,
    )
end

@test success(runmpi(joinpath(@__DIR__, "basic.jl"), ntasks = 2))
