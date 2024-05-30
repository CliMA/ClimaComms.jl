import ..ClimaComms

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
