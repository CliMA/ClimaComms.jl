"""
    SingletonCommsContext(device = device())

A communications context for single-process runs. All communication
primitives (e.g., [`reduce`](@ref), [`gather`](@ref), [`barrier`](@ref))
are no-ops. [`AbstractCPUDevice`](@ref) and [`CUDADevice`](@ref) device
options are currently supported.

# Fields
- `device`: the [`AbstractDevice`](@ref) on which computations run.
"""
struct SingletonCommsContext{D <: AbstractDevice} <: AbstractCommsContext
    device::D
end

SingletonCommsContext() = SingletonCommsContext(device())

device(ctx::SingletonCommsContext) = ctx.device

init(::SingletonCommsContext) = (1, 1)

mypid(::SingletonCommsContext) = 1
iamroot(::SingletonCommsContext) = true
nprocs(::SingletonCommsContext) = 1
barrier(::SingletonCommsContext) = nothing
abort(::SingletonCommsContext, status::Int) = exit(status)
# Copy array buffers so that results do not alias inputs, matching the
# MPI methods, which return newly allocated arrays.
unalias(x::AbstractArray) = copy(x)
unalias(x) = x
reduce(::SingletonCommsContext, val, op) = unalias(val)
gather(::SingletonCommsContext, array) = unalias(array)
allreduce(::SingletonCommsContext, sendbuf, op) = unalias(sendbuf)
bcast(::SingletonCommsContext, object) = object

function reduce!(::SingletonCommsContext, sendbuf, recvbuf, op)
    copyto!(recvbuf, sendbuf)
    return nothing
end
function reduce!(::SingletonCommsContext, sendrecvbuf, op)
    return nothing
end

function allreduce!(::SingletonCommsContext, sendbuf, recvbuf, op)
    copyto!(recvbuf, sendbuf)
    return nothing
end
function allreduce!(::SingletonCommsContext, sendrecvbuf, op)
    return nothing
end

"""
    SingletonGraphContext(context::SingletonCommsContext)

A graph context for single-process runs; [`start`](@ref),
[`progress`](@ref), and [`finish`](@ref) are no-ops.
"""
struct SingletonGraphContext <: AbstractGraphContext
    context::SingletonCommsContext
end

graph_context(ctx::SingletonCommsContext, args...) = SingletonGraphContext(ctx)

start(gctx::SingletonGraphContext) = nothing
progress(gctx::SingletonGraphContext) = nothing
finish(gctx::SingletonGraphContext) = nothing

function Base.summary(io::IO, ctx::SingletonCommsContext)
    println(io, "Context: $(nameof(typeof(ctx)))")
    println(io, "Device: $(typeof(device(ctx)))")
end
