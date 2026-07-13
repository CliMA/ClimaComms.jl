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

# Hint that a missing backend package is the likely cause when a
# `ClimaComms` function has no method for a `CUDADevice` or an
# `MPICommsContext` (its methods live in an extension that is not loaded).
function _backend_hint(io, exc, argtypes, _)
    parentmodule(exc.f) === ClimaComms || return nothing
    if !cuda_ext_is_loaded() && any(T -> T <: CUDADevice, argtypes)
        print(
            io,
            "\n\nA `CUDADevice` argument is involved, but CUDA.jl is not \
             loaded. Load it with `import CUDA` (or \
             `ClimaComms.@import_required_backends`) to enable the \
             CUDA backend.",
        )
    elseif !mpi_ext_is_loaded() && exc.f === MPICommsContext
        print(
            io,
            "\n\nAn `MPICommsContext` is being constructed, but MPI.jl is \
             not loaded. Load it with `import MPI` (or \
             `ClimaComms.@import_required_backends`) to enable the \
             MPI backend.",
        )
    end
    return nothing
end

function __init__()
    Base.Experimental.register_error_hint(_backend_hint, MethodError)
end

end # module
