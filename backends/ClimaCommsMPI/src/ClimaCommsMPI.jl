module ClimaCommsMPI

using ClimaComms
import ClimaComms.Neighbor

using KernelAbstractions
using MPI

struct MPICommsContext <: ClimaComms.AbstractCommsContext
    mpicomm::MPI.Comm
end
MPICommsContext() = MPICommsContext(MPI.COMM_WORLD)

function ClimaComms.init(ctx::MPICommsContext)
    if !MPI.Initialized()
        MPI.Init()
    end

    return ClimaComms.mypid(ctx), ClimaComms.nprocs(ctx)
end

ClimaComms.mypid(ctx::MPICommsContext) = MPI.Comm_rank(ctx.mpicomm) + 1
ClimaComms.iamroot(ctx::MPICommsContext) = ClimaComms.mypid(ctx) == 1
ClimaComms.nprocs(ctx::MPICommsContext) = MPI.Comm_size(ctx.mpicomm)

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


# We could probably do something fancier here?
# Would need to be careful as there is no guarantee that all ranks will call
# finalizers at the same time.
const TAG = Ref(Cint(0))
function newtag(ctx::MPICommsContext)
    TAG[] = tag = mod(TAG[] + 1, MPI.MPI_TAG_UB)
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
    ctx::MPICommsContext
    tag::Cint
    send_bufs::Vector{MPI.Buffer}
    send_ranks::Vector{Cint}
    send_reqs::Vector{MPI.Request}
    recv_bufs::Vector{MPI.Buffer}
    recv_ranks::Vector{Cint}
    recv_reqs::Vector{MPI.Request}
end

function ClimaComms.graph_context(ctx::MPICommsContext, send_array, send_lengths, send_pids, recv_array, recv_lengths, recv_pids)
    @assert length(send_pids) == length(send_lengths)
    @assert length(recv_pids) == length(recv_lengths)

    tag = newtag(ctx)

    send_bufs = MPI.Buffer[]
    total_len = 0
    for len in send_lengths
        buf = MPI.Buffer(view(send_array, total_len+1:total_len+len))
        push!(send_bufs, buf)
        total_len += len
    end
    send_ranks = Cint[pid - 1 for pid in send_pids]
    send_reqs = MPI.Request[MPI.REQUEST_NULL for _ in send_ranks]

    recv_bufs = MPI.Buffer[]
    total_len = 0
    for len in recv_lengths
        buf = MPI.Buffer(view(recv_array, total_len+1:total_len+len))
        push!(recv_bufs, buf)
        total_len += len
    end
    recv_ranks = Cint[pid - 1 for pid in recv_pids]
    recv_reqs = MPI.Request[MPI.REQUEST_NULL for _ in recv_ranks]

    MPISendRecvGraphContext(ctx, tag, send_bufs, send_ranks, send_reqs, recv_bufs, recv_ranks, recv_reqs)
end



function ClimaComms.start(ghost::MPISendRecvGraphContext; dependencies = nothing)
    if !all(r -> r == MPI.REQUEST_NULL, ghost.recv_reqs)
        error("Must finish() before next start()")
    end
    # post receives
    for n in 1:length(ghost.recv_bufs)
        ghost.recv_reqs[n] = MPI.Irecv!(ghost.recv_bufs[n], ghost.recv_ranks[n], ghost.tag, ghost.ctx.mpicomm)
    end
    # post sends
    for n in 1:length(ghost.send_bufs)
        ghost.send_reqs[n] = MPI.Isend(ghost.send_bufs[n], ghost.send_ranks[n], ghost.tag, ghost.ctx.mpicomm)
    end
end

function ClimaComms.progress(ghost::MPISendRecvGraphContext)
    MPI.Iprobe(MPI.MPI_ANY_SOURCE, ghost.tag, ghost.ctx.mpicomm)
end

function ClimaComms.finish(ghost::MPISendRecvGraphContext; dependencies = nothing)
    # wait on previous receives
    MPI.Waitall!(ghost.recv_reqs)
    # ensure that sends have completed
    # TODO: these could be moved to start()? but we would need to add a finalizer to make sure they complete.
    MPI.Waitall!(ghost.send_reqs)
end


end # module
