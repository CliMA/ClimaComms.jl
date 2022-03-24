"""
    SingletonCommsContext()

A singleton communications context, which is used for single-process runs.
"""
struct SingletonCommsContext <: AbstractCommsContext
end

init(::SingletonCommsContext) = (1,1)

mypid(::SingletonCommsContext) = 1
iamroot(::SingletonCommsContext) = true
nprocs(::SingletonCommsContext) = 1
barrier(::SingletonCommsContext) = nothing
reduce(::SingletonCommsContext, val, op) = val
gather(::SingletonCommsContext, array) = array

struct SingletonGraphContext <: AbstractGraphContext
    context::SingletonCommsContext
end

function graph_context(ctx::SingletonCommsContext,
    send_array, send_lengths, send_pids, recv_array, recv_lengths, recv_pids)

    @assert isempty(send_array)
    @assert isempty(send_lengths)
    @assert isempty(send_pids)
    @assert isempty(recv_array)
    @assert isempty(recv_lengths)
    @assert isempty(recv_pids)
    return SingletonGraphContext(ctx)
end

start(gctx::SingletonGraphContext) = nothing
progress(gctx::SingletonGraphContext) = nothing
finish(gctx::SingletonGraphContext) = nothing
