export import_required_backends

function mpi_required()
    return context_type() == :MPICommsContext
end

function cuda_required()
    return device() isa CUDADevice
end

"""
    ClimaComms.@import_required_backends

If the desired context is MPI (as determined by `ClimaComms.context()`), try loading MPI.jl.
If the desired device is CUDA (as determined by `ClimaComms.device()`), try loading CUDA.jl.
"""
macro import_required_backends()
    return quote
        @static if $mpi_required()
            try
                import MPI
            catch
                error(
                    "Cannot load MPI.jl. Make sure it is included in your environment stack.",
                )
            end
        end
        @static if $cuda_required()
            try
                import CUDA
            catch
                error(
                    "Cannot load CUDA.jl. Make sure it is included in your environment stack.",
                )
            end
        end
    end
end
