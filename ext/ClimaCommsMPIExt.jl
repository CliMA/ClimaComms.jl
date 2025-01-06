module ClimaCommsMPIExt

import MPI
import ClimaComms: mpicomm, set_mpicomm!
import ClimaComms

const CLIMA_COMM_WORLD = Ref{typeof(MPI.COMM_WORLD)}()

function set_mpicomm!()
    CLIMA_COMM_WORLD[] = MPI.COMM_WORLD
    return CLIMA_COMM_WORLD[]
end

ClimaComms.mpicomm(::ClimaComms.MPICommsContext) = CLIMA_COMM_WORLD[]

# For backwards compatibility
function Base.getproperty(ctx::ClimaComms.MPICommsContext, sym::Symbol)
    if sym === :mpicomm
        return ClimaComms.mpicomm(ctx)
    else
        return getfield(ctx, sym)
    end
end

function ClimaComms.init(ctx::ClimaComms.MPICommsContext)
    if !MPI.Initialized()
        MPI.Init()
    end
    # TODO: Generalize this to arbitrary accelerators
    if ctx.device isa ClimaComms.CUDADevice
        if !MPI.has_cuda()
            error(
                "MPI implementation is not built with CUDA-aware interface. If your MPI is not OpenMPI, you have to set JULIA_MPI_HAS_CUDA to `true`",
            )
        end
        # assign GPUs based on local rank
        local_comm = MPI.Comm_split_type(
            mpicomm(ctx),
            MPI.COMM_TYPE_SHARED,
            MPI.Comm_rank(mpicomm(ctx)),
        )
        ClimaComms._assign_device(ctx.device, MPI.Comm_rank(local_comm))
        MPI.free(local_comm)
    end
    return ClimaComms.mypid(ctx), ClimaComms.nprocs(ctx)
end

ClimaComms.device(ctx::ClimaComms.MPICommsContext) = ctx.device

ClimaComms.mypid(ctx::ClimaComms.MPICommsContext) =
    MPI.Comm_rank(mpicomm(ctx)) + 1
ClimaComms.iamroot(ctx::ClimaComms.MPICommsContext) = ClimaComms.mypid(ctx) == 1
ClimaComms.nprocs(ctx::ClimaComms.MPICommsContext) = MPI.Comm_size(mpicomm(ctx))

ClimaComms.barrier(ctx::ClimaComms.MPICommsContext) = MPI.Barrier(mpicomm(ctx))

ClimaComms.reduce(ctx::ClimaComms.MPICommsContext, val, op) =
    MPI.Reduce(val, op, 0, mpicomm(ctx))

ClimaComms.reduce!(ctx::ClimaComms.MPICommsContext, sendbuf, recvbuf, op) =
    MPI.Reduce!(sendbuf, recvbuf, op, mpicomm(ctx); root = 0)

ClimaComms.reduce!(ctx::ClimaComms.MPICommsContext, sendrecvbuf, op) =
    MPI.Reduce!(sendrecvbuf, op, mpicomm(ctx); root = 0)

ClimaComms.allreduce(ctx::ClimaComms.MPICommsContext, sendbuf, op) =
    MPI.Allreduce(sendbuf, op, mpicomm(ctx))

ClimaComms.allreduce!(ctx::ClimaComms.MPICommsContext, sendbuf, recvbuf, op) =
    MPI.Allreduce!(sendbuf, recvbuf, op, mpicomm(ctx))

ClimaComms.allreduce!(ctx::ClimaComms.MPICommsContext, sendrecvbuf, op) =
    MPI.Allreduce!(sendrecvbuf, op, mpicomm(ctx))

ClimaComms.bcast(ctx::ClimaComms.MPICommsContext, object) =
    MPI.bcast(object, mpicomm(ctx); root = 0)

function ClimaComms.gather(ctx::ClimaComms.MPICommsContext, array)
    dims = size(array)
    lengths = MPI.Gather(dims[end], 0, mpicomm(ctx))
    if ClimaComms.iamroot(ctx)
        dimsout = (dims[1:(end - 1)]..., sum(lengths))
        arrayout = similar(array, dimsout)
        recvbuf = MPI.VBuffer(arrayout, lengths .* prod(dims[1:(end - 1)]))
    else
        recvbuf = nothing
    end
    MPI.Gatherv!(array, recvbuf, 0, mpicomm(ctx))
end

ClimaComms.abort(ctx::ClimaComms.MPICommsContext, status::Int) =
    MPI.Abort(mpicomm(ctx), status)


# We could probably do something fancier here?
# Would need to be careful as there is no guarantee that all ranks will call
# finalizers at the same time.
const TAG = Ref(Cint(0))
function newtag(ctx::ClimaComms.MPICommsContext)
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
mutable struct MPISendRecvGraphContext <: ClimaComms.AbstractGraphContext
    ctx::ClimaComms.MPICommsContext
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
struct MPIPersistentSendRecvGraphContext <: ClimaComms.AbstractGraphContext
    ctx::ClimaComms.MPICommsContext
    tag::Cint
    send_bufs::Vector{MPI.Buffer}
    send_ranks::Vector{Cint}
    send_reqs::MPI.UnsafeMultiRequest
    recv_bufs::Vector{MPI.Buffer}
    recv_ranks::Vector{Cint}
    recv_reqs::MPI.UnsafeMultiRequest
end

function graph_context(
    ctx::ClimaComms.MPICommsContext,
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
                mpicomm(ctx),
                recv_reqs[n];
                source = recv_ranks[n],
                tag = tag,
            )
        end
        # Allocate a persistent send request
        for n in 1:length(send_bufs)
            MPI.Send_init(
                send_bufs[n],
                mpicomm(ctx),
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

ClimaComms.graph_context(
    ctx::ClimaComms.MPICommsContext,
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

function ClimaComms.start(
    ghost::MPISendRecvGraphContext;
    dependencies = nothing,
)
    if !all(MPI.isnull, ghost.recv_reqs)
        error("Must finish() before next start()")
    end
    # post receives
    for n in 1:length(ghost.recv_bufs)
        MPI.Irecv!(
            ghost.recv_bufs[n],
            ghost.recv_ranks[n],
            ghost.tag,
            mpicomm(ghost.ctx),
            ghost.recv_reqs[n],
        )
    end
    # post sends
    for n in 1:length(ghost.send_bufs)
        MPI.Isend(
            ghost.send_bufs[n],
            ghost.send_ranks[n],
            ghost.tag,
            mpicomm(ghost.ctx),
            ghost.send_reqs[n],
        )
    end
end

function ClimaComms.start(
    ghost::MPIPersistentSendRecvGraphContext;
    dependencies = nothing,
)
    MPI.Startall(ghost.recv_reqs) # post receives
    MPI.Startall(ghost.send_reqs) # post sends
end

function ClimaComms.progress(
    ghost::Union{MPISendRecvGraphContext, MPIPersistentSendRecvGraphContext},
)
    if isdefined(MPI, :MPI_ANY_SOURCE) # < v0.20
        MPI.Iprobe(MPI.MPI_ANY_SOURCE, ghost.tag, mpicomm(ghost.ctx))
    else # >= v0.20
        MPI.Iprobe(MPI.ANY_SOURCE, ghost.tag, mpicomm(ghost.ctx))
    end
end

function ClimaComms.finish(
    ghost::Union{MPISendRecvGraphContext, MPIPersistentSendRecvGraphContext};
    dependencies = nothing,
)
    # wait on previous receives
    MPI.Waitall(ghost.recv_reqs)
    # ensure that sends have completed
    # TODO: these could be moved to start()? but we would need to add a finalizer to make sure they complete.
    MPI.Waitall(ghost.send_reqs)
end

end
