module ClimaCommsMPI

using ClimaComms
import ClimaComms.Neighbor

using KernelAbstractions
using MPI

struct MPICommsContext <: ClimaComms.AbstractCommsContext
    mpicomm::MPI.Comm
    neighbors::Dict{Int, ClimaComms.Neighbor}
    neighbor_pids::Tuple{Vararg{Int}} # we could just use keys(neighbors), but MultiEvents require a Tuple
    recv_reqs::Vector{MPI.Request}
    send_reqs::Vector{MPI.Request}

    function MPICommsContext(
        neighbors::Dict{Int, ClimaComms.Neighbor};
        mpicomm = MPI.COMM_WORLD,
    )
        neighbor_pids = tuple(keys(neighbors)...)
        nneighbors = length(neighbor_pids)
        new(
            mpicomm,
            neighbors,
            neighbor_pids,
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
    return ClimaComms.Neighbor{B}(send_buf, recv_buf)
end

function ClimaComms.init(::Type{MPICommsContext})
    if !MPI.Initialized()
        MPI.Init()
    end

    return ClimaComms.mypid(MPICommsContext), ClimaComms.nprocs(MPICommsContext)
end

ClimaComms.mypid(::Type{MPICommsContext}) = MPI.Comm_rank(MPI.COMM_WORLD) + 1
ClimaComms.mypid(ctx::MPICommsContext) = MPI.Comm_rank(ctx.mpicomm) + 1
ClimaComms.iamroot(CC::Type{MPICommsContext}) = ClimaComms.mypid(CC) == 1
ClimaComms.iamroot(ctx::MPICommsContext) = ClimaComms.mypid(ctx) == 1
ClimaComms.nprocs(::Type{MPICommsContext}) = MPI.Comm_size(MPI.COMM_WORLD)
ClimaComms.nprocs(ctx::MPICommsContext) = MPI.Comm_size(ctx.mpicomm)
ClimaComms.singlebuffered(::Type{MPICommsContext}) = MPI.has_cuda()
ClimaComms.neighbors(ctx::MPICommsContext) = ctx.neighbors

function ClimaComms.start(ctx::MPICommsContext; dependencies = nothing)
    if !all(r -> r == MPI.REQUEST_NULL, ctx.recv_reqs)
        error("Must finish() before next start()")
    end

    progress = () -> iprobe_and_yield(ctx.mpicomm)

    # start moving staged send data to transfer buffers
    events = map(ctx.neighbor_pids) do pid
        ClimaComms.prepare_transfer!(
            ctx.neighbors[pid].send_buf,
            dependencies = dependencies,
            progress = progress,
        )
    end

    # post receives
    for (n, pid) in enumerate(ctx.neighbor_pids)
        tbuf =
            ClimaComms.get_transfer(ClimaComms.recv_buffer(ctx.neighbors[pid]))
        ctx.recv_reqs[n] = MPI.Irecv!(tbuf, pid - 1, 666, ctx.mpicomm)
    end

    # wait for send data to reach the transfer buffers
    wait(CPU(), MultiEvent(events), progress)

    # post sends
    for (n, pid) in enumerate(ctx.neighbor_pids)
        tbuf =
            ClimaComms.get_transfer(ClimaComms.send_buffer(ctx.neighbors[pid]))
        ctx.send_reqs[n] = MPI.Isend(tbuf, pid - 1, 666, ctx.mpicomm)
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

    # wait on previous receives
    MPI.Waitall!(ctx.recv_reqs)

    # move received data to stage buffers
    events = map(ctx.neighbor_pids) do pid
        ClimaComms.prepare_stage!(
            ClimaComms.recv_buffer(ctx.neighbors[pid]);
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

function ClimaComms.gather(ctx::MPICommsContext, array)
    dims = size(array)
    lengths = MPI.Gather(dims[end], 0, ctx.mpicomm)
    if ClimaComms.iamroot(ctx)
        dimsout = (dims[1:(end - 1)]..., sum(lengths))
        arrayout = similar(array, dimsout)
        recvbuf = MPI.VBuffer(arrayout, lengths .* prod(dims[1:(end - 1)]))
    else
        recvbuf = nothing
    end
    MPI.Gatherv!(array, recvbuf, 0, ctx.mpicomm)
end

ClimaComms.abort(ctx::MPICommsContext, status::Int) =
    MPI.Abort(ctx.mpicomm, status)

end # module
