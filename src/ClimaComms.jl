"""
    ClimaComms

Abstracts the communications interface for the various CliMA components
in order to:
- enable the use of different backends as transports (MPI, SharedArrays,
  etc.), and
- transparently support single or double buffering for GPUs, depending
  on whether the transport has the ability to access GPU memory.

Use one of the `ClimaComms` backends, currently:
- `ClimaCommsMPI` uses MPI-2 asynchronous primitives through `MPI.jl`
  and supports a single buffer if the MPI implementation is CUDA-aware.
- `ClimaCommsSA` uses Julia's `Distributed` capability and
  `SharedArrays.jl` for within-node communication.
"""
module ClimaComms

"""
    AbstractCommsContext

The base type for a communications context. Each backend defines a
concrete subtype of this, an instance of which is constructed by passing
in an array of `Neighbor`s.
"""
abstract type AbstractCommsContext end

"""
    init(::Type{CC}) where {CC <: AbstractCommsContext}

Perform any necessary initialization for the specified backend. Return a
tuple of the processor ID and the number of participating processors.
"""
function init(::Type{CC}) where {CC <: AbstractCommsContext}
    error("No `init` method defined for $CC")
end
init(::Nothing) = nothing

"""
    mypid(::Type{CC}) where {CC <: AbstractCommsContext}

Return the processor ID.
"""
function mypid end
mypid(::Nothing) = 1

"""
    iamroot(::Type{CC}) where {CC <: AbstractCommsContext}

Return `true` if the calling processor is the root processor.
"""
function iamroot end
iamroot(::Nothing) = true

"""
    nprocs(::Type{CC}) where {CC <: AbstractCommsContext}

Return the number of participating processors.
"""
function nprocs end
nprocs(::Nothing) = 1

"""
    singlebuffered(::Type{CC}) where {CC <: AbstractCommsContext}

Returns `true` if communication can be single-buffered.
"""
function singlebuffered end
singlebuffered(::Nothing) = true

"""
    neighbors(ctx::CC) where {CC <: AbstractCommsContext}

Returns a `Dict{Int,Neighbor}` mapping the processor ID to the
`Neighbor` for that processor.
"""
function neighbors end
neighbors(::Nothing) = ()

"""
    start(ctx::CC; kwargs...) where {CC <: AbstractCommsContext}

Initiate communication. The stage areas of all the send `RelayBuffer`s
must be filled before this is called!
"""
function start end
start(::Nothing) = nothing

"""
    progress(ctx::CC) where {CC <: AbstractCommsContext}

Drive communication. Call after `start()` to ensure that communication
proceeds asynchronously.
"""
function progress end
progress(::Nothing) = nothing

"""
    finish(ctx::CC; kwargs...) where {CC <: AbstractCommsContext}

Complete the communications step begun by `start()`. After this returns,
data received from all neighbors will be available in the stage areas of
each neighbor's receive buffer.
"""
function finish end
finish(::Nothing) = nothing

"""
    barrier(ctx::CC) where {CC <: AbstractCommsContext}

Perform a global synchronization across all participating processors.
"""
function barrier end
barrier(::Nothing) = nothing

"""
    reduce(ctx::CC, val, op) where {CC <: AbstractCommsContext}

Perform a reduction across all participating processors, using `op` as
the reduction operator and `val` as this rank's reduction value. Return
the result to the first processor only.
"""
function reduce end
reduce(::Nothing, val, op) = val

"""
    gather(ctx::AbstractCommsContext, array)

Gather an array of values from all processors into a single array,
concattenating along the last dimension.
"""
gather(::Nothing, array) = array

"""
    abort(ctx::CC, status::Int) where {CC <: AbstractCommsContext}

Terminate the caller and all participating processors with the specified
`status`.
"""
function abort end
abort(::Nothing, status::Int) = exit(status)

include("relay_buffers.jl")
include("neighbors.jl")

end # module
