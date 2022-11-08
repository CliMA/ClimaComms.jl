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
concrete subtype of this.
"""
abstract type AbstractCommsContext end

"""
    (pid, nprocs) = init(ctx::AbstractCommsContext)

Perform any necessary initialization for the specified backend. Return a
tuple of the processor ID and the number of participating processors.
"""
function init end

"""
    mypid(ctx::AbstractCommsContext)

Return the processor ID.
"""
function mypid end

"""
    iamroot(ctx::AbstractCommsContext)

Return `true` if the calling processor is the root processor.
"""
function iamroot end

"""
    nprocs(ctx::AbstractCommsContext)

Return the number of participating processors.
"""
function nprocs end


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
    allreduce(ctx::CC, sendbuf, op)

Performs elementwise reduction using the operator `op` on the buffer `sendbuf`, allocating a new array for the result.
`sendbuf` can also be a scalar, in which case `recvbuf` will be a value of the same type.
"""
function allreduce end

"""
    allreduce!(ctx::CC, sendbuf, recvbuf, op)
    allreduce!(ctx::CC, sendrecvbuf, op)

Performs elementwise reduction using the operator `op` on the buffer `sendbuf`, storing the result in the `recvbuf` of all processes in the group.
`Allreduce!` is equivalent to a `Reduce!` operation followed by a `Bcast!`, but can lead to better performance.
If only one `sendrecvbuf` buffer is provided, then the operation is performed in-place.

"""
function allreduce! end

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

"""
    AbstractGraphContext

A context for communicating between processes in a graph.
"""
abstract type AbstractGraphContext end

"""
    graph_context(context::AbstractCommsContext, sendarray, sendpids, sendlengths, recvarray, recvpids, recvlengths)

Construct a communication context for exchanging neighbor data via a graph.

Arguments:
- `context`: the communication context on which to construct the graph context.
- `sendarray`: array containing data to send
- `sendpids`: list of processor IDs to send
- `sendlengths`: list of lengths of data to send to each process ID
- `recvarray`: array to receive data into
- `recvpids`: list of processor IDs to receive from
- `recvlengths`: list of lengths of data to receive from each process ID

This should return an `AbstractGraphContext` object.
"""
function graph_context end


"""
    start(ctx::AbstractGraphContext)

Initiate graph data exchange.
"""
function start end

"""
    progress(ctx::AbstractGraphContext)

Drive communication. Call after `start` to ensure that communication
proceeds asynchronously.
"""
function progress end

"""
    finish(ctx::AbstractGraphContext)

Complete the communications step begun by `start()`. After this returns,
data received from all neighbors will be available in the stage areas of
each neighbor's receive buffer.
"""
function finish end

include("singleton.jl")

end # module
