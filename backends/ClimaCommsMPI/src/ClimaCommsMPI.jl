module ClimaCommsMPI

using ClimaComms
import ClimaComms.Neighbor

using KernelAbstractions
using MPI

struct MPICommsContext <: ClimaComms.AbstractCommsContext
    mpicomm::MPI.Comm
    neighbors::Vector{ClimaComms.Neighbor}
    recv_reqs::Vector{MPI.Request}
    send_reqs::Vector{MPI.Request}

    function MPICommsContext(
        neighbors::Vector{ClimaComms.Neighbor};
        mpicomm = MPI.COMM_WORLD,
    )
        nneighbors = length(neighbors)
        new(
            mpicomm,
            neighbors,
            fill(MPI.REQUEST_NULL, nneighbors),
            fill(MPI.REQUEST_NULL, nneighbors),
        )
    end
end

function Neighbor(
    ::Type{MPICommsContext},
    pid,
    AT,
    FT,
    send_dims,
    recv_dims = send_dims,
)
    if pid > 0
        if AT <: Array || singlebuffered(CC)
            kind = ClimaComms.SingleRelayBuffer
        else
            kind = ClimaComms.DoubleRelayBuffer
        end
        send_buf = ClimaComms.RelayBuffer{FT}(AT, kind, send_dims...)
        recv_buf = ClimaComms.RelayBuffer{FT}(AT, kind, recv_dims...)
        B = ClimaComms.RelayBuffer
    else
        send_buf = recv_buf = missing
        B = Missing
    end
    ClimaComms.Neighbor{B}(pid, send_buf, recv_buf)
end

function ClimaComms.init(::Type{MPICommsContext})
    if !MPI.Initialized()
        MPI.Init()
        atexit() do
            MPI.Finalize()
        end
    end

    return ClimaComms.mypid(MPICommsContext), ClimaComms.nprocs(MPICommsContext)
end

ClimaComms.mypid(::Type{MPICommsContext}) = MPI.Comm_rank(MPI.COMM_WORLD) + 1
ClimaComms.mypid(ctx::MPICommsContext) = MPI.Comm_rank(ctx.mpicomm) + 1
ClimaComms.iamroot(CC::Type{MPICommsContext}) = ClimaComms.mypid(CC) == 1
ClimaComms.nprocs(::Type{MPICommsContext}) = MPI.Comm_size(MPI.COMM_WORLD)
ClimaComms.nprocs(ctx::MPICommsContext) = MPI.Comm_size(ctx.mpicomm)
ClimaComms.singlebuffered(::Type{MPICommsContext}) = MPI.has_cuda()

function ClimaComms.start(ctx::MPICommsContext; dependencies = nothing)
    if !all(r -> r == MPI.REQUEST_NULL, ctx.recv_reqs)
        error("Must finish() before next start()")
    end

    progress = () -> iprobe_and_yield(ctx.mpicomm)
    nneighbors = length(ctx.neighbors)

    # start moving staged send data to transfer buffers
    events = ntuple(nneighbors) do n
        ClimaComms.prepare_transfer!(
            ctx.neighbors[n].send_buf,
            dependencies = dependencies,
            progress = progress,
        )
    end

    # post receives
    for n in 1:nneighbors
        tbuf = ClimaComms.get_transfer(ClimaComms.recv_buffer(ctx.neighbors[n]))
        ctx.recv_reqs[n] = MPI.Irecv!(
            tbuf,
            ClimaComms.pid(ctx.neighbors[n]) - 1,
            666,
            ctx.mpicomm,
        )
    end

    # wait for send data to reach the transfer buffers
    wait(CPU(), MultiEvent(events), progress)

    # post sends
    for n in 1:nneighbors
        tbuf = ClimaComms.get_transfer(ClimaComms.send_buffer(ctx.neighbors[n]))
        ctx.send_reqs[n] = MPI.Isend(
            tbuf,
            ClimaComms.pid(ctx.neighbors[n]) - 1,
            666,
            ctx.mpicomm,
        )
    end
end

function ClimaComms.progress(ctx::MPICommsContext)
    MPI.Iprobe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG, ctx.mpicomm)
end

function ClimaComms.finish(ctx::MPICommsContext; dependencies = nothing)
    if any(r -> r == MPI.REQUEST_NULL, ctx.recv_reqs)
        error("Must start() before finish()")
    end

    progress = () -> iprobe_and_yield(ctx.mpicomm)
    nneighbors = length(ctx.neighbors)

    # wait on previous receives
    MPI.Waitall!(ctx.recv_reqs)

    # move received data to stage buffers
    events = ntuple(nneighbors) do n
        ClimaComms.prepare_stage!(
            ClimaComms.recv_buffer(ctx.neighbors[n]);
            dependencies = dependencies,
            progress = progress,
        )
    end

    # ensure that sends have completed
    MPI.Waitall!(ctx.send_reqs)

    return MultiEvent(events)
end

function iprobe_and_yield(mpicomm)
    MPI.Iprobe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG, mpicomm)
    yield()
end

ClimaComms.barrier(ctx::MPICommsContext) = MPI.Barrier(ctx.mpicomm)

ClimaComms.reduce(ctx::MPICommsContext, val, op) =
    MPI.Reduce(val, op, 0, ctx.mpicomm)

ClimaComms.abort(ctx::MPICommsContext, status::Int) =
    MPI.Abort(ctx.mpicomm, status)

end # module
