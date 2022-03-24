using MPI
using Test

function runmpi(file; ntasks = 1)
    # Some MPI runtimes complain if more resources are requested
    # than available.
    if MPI.MPI_LIBRARY == MPI.OpenMPI
        oversubscribe = `--oversubscribe`
    else
        oversubscribe = ``
    end
    MPI.mpiexec() do cmd
        Base.run(
            `$cmd $oversubscribe -np $ntasks $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $file`;
            wait = true,
        )
        true
    end
end
#=
@testset "Stencil" begin
    runmpi(joinpath(@__DIR__, "mpi_stencil.jl"), ntasks = 4)
end
=#