using MPI, Test

function runmpi(file; ntasks = 1)
    # Some MPI runtimes complain if more resources are requested
    # than available.
    MPI.mpiexec() do cmd
        Base.run(
            `$cmd -n $ntasks $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $file`;
            wait = true,
        )
        true
    end
end

@testset "Stencil" begin
    runmpi(joinpath(@__DIR__, "mpi_stencil.jl"), ntasks = 4)
end
