using MPI

"""
    MPICommsContext()
    MPICommsContext(device)
    MPICommsContext(device, comm)

A MPI communications context, used for distributed runs.
[`CPUDevice`](@ref) and [`CUDADevice`](@ref) device options are currently supported.
"""
struct MPICommsContext{D <: AbstractDevice, C <: MPI.Comm} <:
       AbstractCommsContext
    device::D
    mpicomm::C
end
MPICommsContext(device = device()) = MPICommsContext(device, MPI.COMM_WORLD)

function init(ctx::MPICommsContext)
    if !MPI.Initialized()
        MPI.Init()
    end
    return mypid(ctx), nprocs(ctx)
end

mypid(ctx::MPICommsContext) = MPI.Comm_rank(ctx.mpicomm) + 1
iamroot(ctx::MPICommsContext) = mypid(ctx) == 1
nprocs(ctx::MPICommsContext) = MPI.Comm_size(ctx.mpicomm)

barrier(ctx::MPICommsContext) = MPI.Barrier(ctx.mpicomm)

reduce(ctx::MPICommsContext, val, op) = MPI.Reduce(val, op, 0, ctx.mpicomm)

allreduce(ctx::MPICommsContext, sendbuf, op) =
    MPI.Allreduce(sendbuf, op, ctx.mpicomm)

allreduce!(ctx::MPICommsContext, sendbuf, recvbuf, op) =
    MPI.Allreduce!(sendbuf, recvbuf, op, ctx.mpicomm)

allreduce!(ctx::MPICommsContext, sendrecvbuf, op) =
    MPI.Allreduce!(sendrecvbuf, op, ctx.mpicomm)

function gather(ctx::MPICommsContext, array)
    dims = size(array)
    lengths = MPI.Gather(dims[end], 0, ctx.mpicomm)
    if iamroot(ctx)
        dimsout = (dims[1:(end - 1)]..., sum(lengths))
        arrayout = similar(array, dimsout)
        recvbuf = MPI.VBuffer(arrayout, lengths .* prod(dims[1:(end - 1)]))
    else
        recvbuf = nothing
    end
    MPI.Gatherv!(array, recvbuf, 0, ctx.mpicomm)
end

abort(ctx::MPICommsContext, status::Int) = MPI.Abort(ctx.mpicomm, status)


# We could probably do something fancier here?
# Would need to be careful as there is no guarantee that all ranks will call
# finalizers at the same time.
const TAG = Ref(Cint(0))
function newtag(ctx::MPICommsContext)
    TAG[] = tag = mod(TAG[] + 1, 32767) # TODO: this should query MPI_TAG_UB attribute (https://github.com/JuliaParallel/MPI.jl/pull/551)
    if tag == 0
        @warn("MPICommsMPI: tag overflow")
    end
    return tag
end

"""
    MPISendRecvGraphContext

A simple ghost buffer implementation using MPI `Isend`/`Irecv` operations.
"""
mutable struct MPISendRecvGraphContext <: AbstractGraphContext
    ctx::MPICommsContext
    tag::Cint
    send_bufs::Vector{MPI.Buffer}
    send_ranks::Vector{Cint}
    send_reqs::MPI.UnsafeMultiRequest
    recv_bufs::Vector{MPI.Buffer}
    recv_ranks::Vector{Cint}
    recv_reqs::MPI.UnsafeMultiRequest
end

"""
    MPIPersistentSendRecvGraphContext

A simple ghost buffer implementation using MPI persistent send/receive operations.
"""
struct MPIPersistentSendRecvGraphContext <: AbstractGraphContext
    ctx::MPICommsContext
    tag::Cint
    send_bufs::Vector{MPI.Buffer}
    send_ranks::Vector{Cint}
    send_reqs::MPI.UnsafeMultiRequest
    recv_bufs::Vector{MPI.Buffer}
    recv_ranks::Vector{Cint}
    recv_reqs::MPI.UnsafeMultiRequest
end

function graph_context(
    ctx::MPICommsContext,
    send_array,
    send_lengths,
    send_pids,
    recv_array,
    recv_lengths,
    recv_pids,
    ::Type{GCT},
) where {
    GCT <: Union{MPISendRecvGraphContext, MPIPersistentSendRecvGraphContext},
}
    @assert length(send_pids) == length(send_lengths)
    @assert length(recv_pids) == length(recv_lengths)

    tag = newtag(ctx)

    send_bufs = MPI.Buffer[]
    total_len = 0
    for len in send_lengths
        buf = MPI.Buffer(view(send_array, (total_len + 1):(total_len + len)))
        push!(send_bufs, buf)
        total_len += len
    end
    send_ranks = Cint[pid - 1 for pid in send_pids]
    send_reqs = MPI.UnsafeMultiRequest(length(send_ranks))

    recv_bufs = MPI.Buffer[]
    total_len = 0
    for len in recv_lengths
        buf = MPI.Buffer(view(recv_array, (total_len + 1):(total_len + len)))
        push!(recv_bufs, buf)
        total_len += len
    end
    recv_ranks = Cint[pid - 1 for pid in recv_pids]
    recv_reqs = MPI.UnsafeMultiRequest(length(recv_ranks))
    args = (
        ctx,
        tag,
        send_bufs,
        send_ranks,
        send_reqs,
        recv_bufs,
        recv_ranks,
        recv_reqs,
    )
    if GCT == MPIPersistentSendRecvGraphContext
        # Allocate a persistent receive request
        for n in 1:length(recv_bufs)
            MPI.Recv_init(
                recv_bufs[n],
                ctx.mpicomm,
                recv_reqs[n];
                source = recv_ranks[n],
                tag = tag,
            )
        end
        # Allocate a persistent send request
        for n in 1:length(send_bufs)
            MPI.Send_init(
                send_bufs[n],
                ctx.mpicomm,
                send_reqs[n];
                dest = send_ranks[n],
                tag = tag,
            )
        end
        MPIPersistentSendRecvGraphContext(args...)
    else
        MPISendRecvGraphContext(args...)
    end
end

graph_context(
    ctx::MPICommsContext,
    send_array,
    send_lengths,
    send_pids,
    recv_array,
    recv_lengths,
    recv_pids;
    persistent::Bool = true,
) = graph_context(
    ctx,
    send_array,
    send_lengths,
    send_pids,
    recv_array,
    recv_lengths,
    recv_pids,
    persistent ? MPIPersistentSendRecvGraphContext : MPISendRecvGraphContext,
)

function start(ghost::MPISendRecvGraphContext; dependencies = nothing)
    if !all(r -> r == MPI.REQUEST_NULL, ghost.recv_reqs)
        error("Must finish() before next start()")
    end
    # post receives
    for n in 1:length(ghost.recv_bufs)
        MPI.Irecv!(
            ghost.recv_bufs[n],
            ghost.recv_ranks[n],
            ghost.tag,
            ghost.ctx.mpicomm,
            ghost.recv_reqs[n],
        )
    end
    # post sends
    for n in 1:length(ghost.send_bufs)
        MPI.Isend(
            ghost.send_bufs[n],
            ghost.send_ranks[n],
            ghost.tag,
            ghost.ctx.mpicomm,
            ghost.send_reqs[n],
        )
    end
end

function start(ghost::MPIPersistentSendRecvGraphContext; dependencies = nothing)
    MPI.Startall(ghost.recv_reqs) # post receives
    MPI.Startall(ghost.send_reqs) # post sends
end

function progress(
    ghost::Union{MPISendRecvGraphContext, MPIPersistentSendRecvGraphContext},
)
    if isdefined(MPI, :MPI_ANY_SOURCE) # < v0.20
        MPI.Iprobe(MPI.MPI_ANY_SOURCE, ghost.tag, ghost.ctx.mpicomm)
    else # >= v0.20
        MPI.Iprobe(MPI.ANY_SOURCE, ghost.tag, ghost.ctx.mpicomm)
    end
end

function finish(
    ghost::Union{MPISendRecvGraphContext, MPIPersistentSendRecvGraphContext};
    dependencies = nothing,
)
    # wait on previous receives
    MPI.Waitall(ghost.recv_reqs)
    # ensure that sends have completed
    # TODO: these could be moved to start()? but we would need to add a finalizer to make sure they complete.
    MPI.Waitall(ghost.send_reqs)
end
