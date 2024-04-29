import ..ClimaComms

export import_required_backends

function mpi_is_required()
    return context_type() == :MPICommsContext
end

function mpi_ext_is_not_loaded()
    return isnothing(Base.get_extension(ClimaComms, :ClimaCommsMPIExt))
end

function cuda_is_required()
    return device_type() == :CUDADevice
end

function cuda_ext_is_not_loaded()
    return isnothing(Base.get_extension(ClimaComms, :ClimaCommsCUDAExt))
end

"""
    ClimaComms.@import_required_backends

If the desired context is MPI (as determined by `ClimaComms.context()`), try loading MPI.jl.
If the desired device is CUDA (as determined by `ClimaComms.device()`), try loading CUDA.jl.
"""
macro import_required_backends()
    return quote
        @static if $mpi_is_required() && $mpi_ext_is_not_loaded()
            import MPI
            @info "Loaded MPI.jl"
        end
        @static if $cuda_is_required() && $cuda_ext_is_not_loaded()
            import CUDA
            @info "Loaded CUDA.jl"
        end
    end
end
