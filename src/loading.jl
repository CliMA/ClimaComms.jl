import ..ClimaComms

export @import_required_backends

"""
    ClimaComms.mpi_is_required()

Return `true` if `MPI.jl` needs to be loaded, based on the
`CLIMACOMMS_CONTEXT` environment variable. See [`context`](@ref) for more
information.

# Examples
```julia
mpi_is_required() && using MPI
```
"""
mpi_is_required() = context_type() == :MPICommsContext

"""
    ClimaComms.cuda_is_required()

Return `true` if `CUDA.jl` needs to be loaded, based on the
`CLIMACOMMS_DEVICE` environment variable. See [`device`](@ref) for more
information.

# Examples
```julia
cuda_is_required() && using CUDA
```
"""
cuda_is_required() = device_type() == :CUDADevice

"""
    ClimaComms.cuda_ext_is_loaded()

Return `true` if the `ClimaCommsCUDAExt` extension is loaded (i.e., if
`CUDA.jl` has been imported).
"""
cuda_ext_is_loaded() =
    !isnothing(Base.get_extension(ClimaComms, :ClimaCommsCUDAExt))

"""
    ClimaComms.mpi_ext_is_loaded()

Return `true` if the `ClimaCommsMPIExt` extension is loaded (i.e., if
`MPI.jl` has been imported).
"""
mpi_ext_is_loaded() =
    !isnothing(Base.get_extension(ClimaComms, :ClimaCommsMPIExt))

"""
    ClimaComms.@import_required_backends

Import the backend packages required by the runtime configuration: if the
`CLIMACOMMS_CONTEXT` environment variable requests MPI, import `MPI.jl`;
if the `CLIMACOMMS_DEVICE` environment variable requests CUDA, import
`CUDA.jl`. The packages must be available in the active Julia environment.

Add this macro to the top of driver scripts, after `import ClimaComms`, so
that the same script works for any device and context.

!!! warning
    Do not use this macro in library code (i.e., in `src`): it imports
    packages that libraries should not depend on. Only use it in scripts,
    where the environment can be expected to provide the backends.

# Examples
```julia
import ClimaComms
ClimaComms.@import_required_backends
context = ClimaComms.context()
```
"""
macro import_required_backends()
    return quote
        @static if $mpi_is_required()
            @debug "Loading MPI via `import MPI`..."
            import MPI
        end
        @static if $cuda_is_required()
            @debug "Loading CUDA via `import CUDA`..."
            import CUDA
        end
    end
end
