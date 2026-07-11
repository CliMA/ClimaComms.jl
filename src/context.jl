import ..ClimaComms

"""
    ClimaComms.context_type()

Return the type of context requested by the `CLIMACOMMS_CONTEXT`
environment variable, as a `Symbol` (`:SingletonCommsContext` or
`:MPICommsContext`).

Called from [`context`](@ref), which constructs the context instance.
"""
function context_type()
    name = get(ENV, "CLIMACOMMS_CONTEXT", "SINGLETON")
    if name == "MPI"
        return :MPICommsContext
    elseif name == "SINGLETON"
        return :SingletonCommsContext
    else
        error("Invalid context: $name")
    end
end

"""
    ClimaComms.context(device = device())

Construct the communication context specified by the `CLIMACOMMS_CONTEXT`
environment variable.

Allowed values of `CLIMACOMMS_CONTEXT`:
- `SINGLETON` (default): a [`SingletonCommsContext`](@ref), for
  single-process runs;
- `MPI`: an [`MPICommsContext`](@ref), for distributed runs, which requires
  `MPI.jl` to be loaded (see [`@import_required_backends`](@ref)).

The context wraps the given `device`; by default, the device is also read
from an environment variable (see [`device`](@ref)).

# Examples
```julia
context = ClimaComms.context()
device = ClimaComms.device(context)
```
"""
function context(device = device(); target_context = context_type())
    if target_context == :MPICommsContext && !mpi_ext_is_loaded()
        error(
            "Loading MPI.jl is required to use MPICommsContext. You might want to call ClimaComms.@import_required_backends",
        )
    end
    ContextConstructor = getproperty(ClimaComms, target_context)
    return ContextConstructor(device)
end

"""
    AbstractCommsContext

The environment through which processes communicate.

A context wraps an [`AbstractDevice`](@ref) and, for distributed runs, the
information needed for processes to exchange data. Contexts make code
independent of the form of parallelism: communication primitives such as
[`reduce`](@ref), [`gather`](@ref), and [`barrier`](@ref) dispatch on the
context and become no-ops in single-process runs.

Subtypes:
- [`SingletonCommsContext`](@ref): a single process; all communication
  primitives are no-ops.
- [`MPICommsContext`](@ref): distributed runs via MPI.

Use [`context`](@ref) to select a context at runtime from the
`CLIMACOMMS_CONTEXT` environment variable.
"""
abstract type AbstractCommsContext end

"""
    ClimaComms.init(ctx::AbstractCommsContext)

Perform any necessary initialization for the specified backend (e.g.,
initializing MPI and assigning GPUs to MPI ranks). Return a tuple
`(pid, nprocs)` of the process ID and the number of participating
processes.

Call this once, before any other communication operations on `ctx`.
"""
function init end

"""
    ClimaComms.mypid(ctx::AbstractCommsContext)

Return the process ID of the calling process, an integer between 1 and
[`nprocs`](@ref). The root process has `mypid(ctx) == 1`.
"""
function mypid end

"""
    ClimaComms.iamroot(ctx::AbstractCommsContext)

Return `true` if the calling process is the root process (the process with
ID 1).
"""
function iamroot end

"""
    ClimaComms.nprocs(ctx::AbstractCommsContext)

Return the number of participating processes.
"""
function nprocs end


"""
    ClimaComms.barrier(ctx::AbstractCommsContext)

Perform a global synchronization across all participating processes: each
process blocks until every process has reached the barrier.
"""
function barrier end
barrier(::Nothing) = nothing

"""
    ClimaComms.reduce(ctx::AbstractCommsContext, val, op)

Perform a reduction across all participating processes, using `op` as the
reduction operator and `val` as this process's contribution. The result is
only valid on the root process.

See also [`allreduce`](@ref) to make the result available on all
processes.
"""
function reduce end
reduce(::Nothing, val, op) = val

"""
    ClimaComms.reduce!(ctx::AbstractCommsContext, sendbuf, recvbuf, op)
    ClimaComms.reduce!(ctx::AbstractCommsContext, sendrecvbuf, op)

Perform an elementwise reduction across all participating processes, using
`op` as the reduction operator and `sendbuf` as this process's
contribution, and store the result in the root process's `recvbuf`. If a
single `sendrecvbuf` buffer is provided, the reduction is performed
in-place. Return `nothing`.

See also [`allreduce!`](@ref) to make the result available on all
processes.
"""
function reduce! end

"""
    ClimaComms.allreduce(ctx::AbstractCommsContext, sendbuf, op)

Perform an elementwise reduction across all participating processes, using
`op` as the reduction operator and `sendbuf` as this process's
contribution, and return the result in a newly allocated array on every
process. `sendbuf` can also be a scalar, in which case the result is a
value of the same type.
"""
function allreduce end

"""
    ClimaComms.allreduce!(ctx::AbstractCommsContext, sendbuf, recvbuf, op)
    ClimaComms.allreduce!(ctx::AbstractCommsContext, sendrecvbuf, op)

Perform an elementwise reduction across all participating processes, using
`op` as the reduction operator and `sendbuf` as this process's
contribution, and store the result in the `recvbuf` of every process. If a
single `sendrecvbuf` buffer is provided, the reduction is performed
in-place. Return `nothing`.

`allreduce!` is equivalent to [`reduce!`](@ref) followed by
[`bcast`](@ref), but can achieve better performance.
"""
function allreduce! end

"""
    ClimaComms.gather(ctx::AbstractCommsContext, array)

Gather an array from each participating process into a single array on the
root process, concatenating along the last dimension. The arrays must have
the same size on every process except possibly in the last dimension. The
result is only valid on the root process.
"""
gather(::Nothing, array) = array

"""
    ClimaComms.bcast(ctx::AbstractCommsContext, object)

Broadcast `object` from the root process to all other processes, and
return it on every process. The value of `object` on non-root processes is
ignored.
"""
function bcast end

"""
    ClimaComms.abort(ctx::AbstractCommsContext, status::Int)

Terminate the caller and all participating processes with the specified
exit `status`.
"""
function abort end
abort(::Nothing, status::Int) = exit(status)

"""
    AbstractGraphContext

A context for exchanging data between neighboring processes in a graph,
such as the ghost (halo) regions of a domain decomposition.

Construct with [`graph_context`](@ref); exchange data with
[`start`](@ref), [`progress`](@ref), and [`finish`](@ref).
"""
abstract type AbstractGraphContext end

"""
    ClimaComms.graph_context(
        context::AbstractCommsContext,
        sendarray, sendlengths, sendpids,
        recvarray, recvlengths, recvpids,
    )

Construct an [`AbstractGraphContext`](@ref) for exchanging neighbor data
via a graph.

# Arguments
- `context`: the communication context on which to construct the graph
  context.
- `sendarray`: array containing the data to send, ordered by destination
  process.
- `sendlengths`: list of the number of elements to send to each process in
  `sendpids`.
- `sendpids`: list of process IDs to send to.
- `recvarray`: array to receive data into, ordered by source process.
- `recvlengths`: list of the number of elements to receive from each
  process in `recvpids`.
- `recvpids`: list of process IDs to receive from.

# Notes
For [`MPICommsContext`](@ref), the keyword argument `persistent = true`
selects persistent MPI send/receive requests instead of `MPI.Isend` /
`MPI.Irecv!`, which reduces the overhead of repeated exchanges.
"""
function graph_context end


"""
    ClimaComms.start(ctx::AbstractGraphContext)

Initiate the graph data exchange: post the receives and sends for the data
currently in the send buffers.
"""
function start end

"""
    ClimaComms.progress(ctx::AbstractGraphContext)

Drive communication. Call after [`start`](@ref) to ensure that
communication proceeds asynchronously while other work is performed.
"""
function progress end

"""
    ClimaComms.finish(ctx::AbstractGraphContext)

Complete the communication step begun by [`start`](@ref). After this
returns, the data received from all neighbors is available in the receive
buffers.
"""
function finish end
