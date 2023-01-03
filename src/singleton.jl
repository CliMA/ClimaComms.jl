"""
    SingletonCommsContext(device = CPU())

A singleton communications context, used for single-process runs.
"""
struct SingletonCommsContext <: AbstractCommsContext
    device::AbstractDevice
end

SingletonCommsContext(device = CPU()) = SingletonCommsContext(device)

init(::SingletonCommsContext) = (1, 1)

mypid(::SingletonCommsContext) = 1
iamroot(::SingletonCommsContext) = true
nprocs(::SingletonCommsContext) = 1
barrier(::SingletonCommsContext) = nothing
reduce(::SingletonCommsContext, val, op) = val
gather(::SingletonCommsContext, array) = array
allreduce(::SingletonCommsContext, sendbuf, op) = sendbuf
function allreduce!(::SingletonCommsContext, sendbuf, recvbuf, op)
    copyto!(recvbuf, sendbuf)
    return nothing
end
function allreduce!(::SingletonCommsContext, sendrecvbuf, op)
    return nothing
end
struct SingletonGraphContext <: AbstractGraphContext
    context::SingletonCommsContext
end

graph_context(ctx::SingletonCommsContext, kwargs...) =
    SingletonGraphContext(ctx)

start(gctx::SingletonGraphContext) = nothing
progress(gctx::SingletonGraphContext) = nothing
finish(gctx::SingletonGraphContext) = nothing
