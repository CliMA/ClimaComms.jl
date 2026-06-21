import ..ClimaComms

export @import_required_backends

"""
    mpi_is_required()

Returns a Bool indicating if MPI should be loaded, based on the
`ENV["CLIMACOMMS_CONTEXT"]`. See [`ClimaComms.context`](@ref) for
more information.

```julia
mpi_is_required() && using MPI
```
"""
mpi_is_required() = context_type() == :MPICommsContext

"""
    cuda_is_required()

Returns a Bool indicating if CUDA should be loaded, based on the
`ENV["CLIMACOMMS_DEVICE"]`. See [`ClimaComms.device`](@ref) for
more information.

```julia
cuda_is_required() && using CUDA
```
"""
cuda_is_required() = device_type() == :CUDADevice

cuda_ext_is_loaded() =
    !isnothing(Base.get_extension(ClimaComms, :ClimaCommsCUDAExt))

mpi_ext_is_loaded() =
    !isnothing(Base.get_extension(ClimaComms, :ClimaCommsMPIExt))

"""
    ClimaComms.@import_required_backends

If the desired context is MPI (as determined by `ClimaComms.context()`), try loading MPI.jl.
If the desired device is CUDA (as determined by `ClimaComms.device()`), try loading CUDA.jl.
"""
macro import_required_backends()
    return quote
        @static if $mpi_is_required()
            if isnothing(Base.identify_package("MPI"))
                error(
                    """ClimaComms has requested a parallel context, but MPI.jl is not
                 available in your current environment. Most CliMA packages do not come
                 with MPI.jl and acquiring this package it is left to the end users who
                 need it. Add MPI.jl to your environment or use an environment that
                 contains this package.""",
                )
            end

            @debug "Loading MPI via `import MPI`..."
            import MPI
        end
        @static if $cuda_is_required()
            if isnothing(Base.identify_package("CUDA"))
                error(
                    """ClimaComms has requested a CUDA device, but CUDA.jl is not
                 available in your current environment. Most CliMA packages and do not
                 come with CUDA.jl and acquiring this package it is left to the end users
                 who need it. Add CUDA.jl to your environment or use an environment that
                 contains this package.""",
                )
            end

            @debug "Loading CUDA via `import CUDA`..."
            import CUDA
        end
    end
end
