"""
    ClimaComms

Abstract the computing devices and communication contexts used by CliMA
packages, so that the same simulation code can run on a single CPU thread,
on multiple CPU threads, on NVIDIA GPUs, and across MPI processes.

The two central abstractions are:
- [`AbstractDevice`](@ref): the hardware a computation runs on (e.g.,
  [`CPUSingleThreaded`](@ref), [`CUDADevice`](@ref)).
- [`AbstractCommsContext`](@ref): the environment through which processes
  communicate (e.g., [`SingletonCommsContext`](@ref),
  [`MPICommsContext`](@ref)).

Devices and contexts are selected at runtime, typically from the
`CLIMACOMMS_DEVICE` and `CLIMACOMMS_CONTEXT` environment variables via
[`device`](@ref) and [`context`](@ref). Backend-specific implementations
(CUDA, MPI) live in package extensions and are loaded with
[`@import_required_backends`](@ref).
"""
module ClimaComms

include("devices.jl")
include("context.jl")
include("singleton.jl")
include("mpi.jl")
include("loading.jl")
include("adapt.jl")
include("logging.jl")

end # module
