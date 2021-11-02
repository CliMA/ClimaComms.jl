"""
    MPICommsType

The communications type used for MPI. Construct it by specifying the
ranks with which data must be exchanged as well as a receive and a
send `RelayBuffer` for each neighbor.
"""
struct MPICommsType <: AbstractCommsType
    mpi_comm::MPI.Comm
    neighbors::Vector{Int}
    recv_buffers::Vector{RelayBuffer}
    recv_reqs::Vector{MPI.Request}
    send_buffers::Vector{RelayBuffer}
    send_reqs::Vector{MPI.Request}

    function MPICommsType(
        neighbors::Vector,
        recv_buffers,
        send_buffers,
        mpi_comm = MPI.COMM_WORLD,
    )
        nneighbors = length(neighbors)
        new(
            mpi_comm,
            neighbors,
            recv_buffers,
            fill(MPI.REQUEST_NULL, nneighbors),
            send_buffers,
            fill(MPI.REQUEST_NULL, nneighbors),
        )
    end
end

"""
    init(::Type{MPICommsType})

Initializes MPI and returns a tuple of the MPI rank and the number
of ranks in the job.
"""
function init(::Type{MPICommsType})
    if MPI.Initialized()
        @warn "MPI has already been initialized."
    else
        MPI.Init()
        atexit() do
            MPI.Finalize()
        end
    end

    return MPI.Comm_rank(MPI.COMM_WORLD), MPI.Comm_size(MPI.COMM_WORLD)
end

"""
    start(ctx::MPICommsType; dependencies = nothing)

Initiate communication. The stage areas of all the send `RelayBuffer`s
must be filled before this is called! Posts asynchronous receives for
each neighbor, followed by asynchronous sends for each neighbor.
"""
function start(ctx::MPICommsType; dependencies = nothing)
    if !all(r -> r == MPI.REQUEST_NULL, ctx.recv_reqs)
        error("Must finish() before next start()")
    end

    progress = () -> iprobe_and_yield(ctx.mpi_comm)
    nneighbors = length(ctx.neighbors)

    # start moving staged send data to transfer buffers
    events = ()
    for n in 1:nneighbors
        event = prepare_transfer!(
            ctx.send_buffers[n];
            dependencies = dependencies,
            progress = progress,
        )
        events = (event, events...)
    end

    # post receives
    for n in 1:nneighbors
        tbuf = get_transfer(ctx.recv_buffers[n])
        ctx.recv_reqs[n] = MPI.Irecv!(tbuf, ctx.neighbors[n], 666, ctx.mpi_comm)
    end

    # wait for send data to reach the transfer buffers
    wait(CPU(), MultiEvent(events), progress)

    # post sends
    for n in 1:nneighbors
        tbuf = get_transfer(ctx.send_buffers[n])
        ctx.send_reqs[n] = MPI.Isend(tbuf, ctx.neighbors[n], 666, ctx.mpi_comm)
    end
end

"""
    progress(ctx::MPICommsType)

Calls `MPI.Iprobe()` to drive the MPI progress engine. May be called
after `start()` to ensure that communication proceeds asynchronously.
"""
function progress(ctx::MPICommsType)
    MPI.Iprobe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG, ctx.mpi_comm)
end

"""
    finish(ctx::MPICommsType; dependencies = nothing)

Complete the communications step begun by `start()`. Completes the
receives posted in `start()` and moves the received data into the
stage areas of the receive buffers.
"""
function finish(ctx::MPICommsType; dependencies = nothing)
    if any(r -> r == MPI.REQUEST_NULL, ctx.recv_reqs)
        error("Must start() before finish()")
    end

    progress = () -> iprobe_and_yield(ctx.mpi_comm)

    nneighbors = length(ctx.neighbors)

    # wait on previous receives
    MPI.Waitall!(ctx.recv_reqs)
    fill!(ctx.recv_reqs, MPI.REQUEST_NULL)

    # move received data to stage buffers
    events = ()
    for n in 1:nneighbors
        event = prepare_stage!(
            ctx.recv_buffers[n];
            dependencies = dependencies,
            progress = progress,
        )
        events = (event, events...)
    end

    # ensure that sends have completed
    MPI.Waitall!(ctx.send_reqs)
    fill!(ctx.send_reqs, MPI.REQUEST_NULL)

    return MultiEvent(events)
end

function iprobe_and_yield(mpi_comm)
    MPI.Iprobe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG, mpi_comm)
    yield()
end
