module ClimaCommsSA

using ClimaComms
import ClimaComms: Neighbor

using Distributed
using KernelAbstractions
using SharedArrays

const ctx_chnl = Ref(RemoteChannel())
get_ctx_chnl() = ctx_chnl[]
const nbr_chnls = Dict{Int, RemoteChannel}()
get_nbr_chnl(nbr) = nbr_chnls[nbr]

mutable struct CCBarrier
    sense::SharedArray{Bool, 1}
    worker_sense::SharedArray{Bool, 1}
    function CCBarrier()
        sense = SharedArray{Bool}((1,), init = (sa) -> sa[1] = false)
        worker_sense =
            SharedArray{Bool}((nprocs(),), init = (sa) -> fill!(sa, false))
        new(sense, worker_sense)
    end
end

mutable struct CCReducer{FT}
    sense::SharedArray{Bool, 1}
    worker_sense::SharedArray{Bool, 1}
    vals::SharedArray{FT, 1}
    function CCReducer{FT}() where {FT}
        sense = SharedArray{Bool}((1,), init = (sa) -> sa[1] = false)
        worker_sense =
            SharedArray{Bool}((nprocs(),), init = (sa) -> fill!(sa, false))
        vals = SharedArray{FT}((nprocs(),), init = (sa) -> fill!(sa, zero(FT)))
        new(sense, worker_sense, vals)
    end
end

struct SACommsContext <: ClimaComms.AbstractCommsContext
    neighbors::Vector{ClimaComms.Neighbor}
    barrier::CCBarrier
    reducer_f64::CCReducer
    reducer_f32::CCReducer

    function SACommsContext(neighbors::Vector{ClimaComms.Neighbor})
        if ClimaComms.iamroot(ClimaCommsSA.SACommsContext)
            barr = CCBarrier()
            red64 = CCReducer{Float64}()
            red32 = CCReducer{Float32}()
            if nprocs() > 1
                for i in 1:nprocs()
                    put!(ClimaCommsSA.get_ctx_chnl(), (barr, red64, red32))
                end
            end
        else
            (barr, red64, red32) = take!(ClimaCommsSA.get_ctx_chnl())
        end
        new(neighbors, barr, red64, red32)
    end
end

function Neighbor(
    ::Type{SACommsContext},
    pid,
    AT,
    FT,
    send_dims,
    recv_dims = send_dims,
)
    @assert AT <: Array # until support for CuArray is added
    if pid <= 0
        return ClimaComms.Neighbor{Missing}(pid, missing, missing)
    end

    if ClimaComms.mypid(SACommsContext) < pid
        nbr_chnls[pid] = RemoteChannel()
        send_buf = SharedArray{FT}(
            send_dims,
            init = (sa) -> fill!(sa, zero(FT)),
            pids = [ClimaComms.mypid(SACommsContext), pid],
        )
        recv_buf = SharedArray{FT}(
            recv_dims,
            init = (sa) -> fill!(sa, zero(FT)),
            pids = [ClimaComms.mypid(SACommsContext), pid],
        )
        put!(ClimaCommsSA.get_nbr_chnl(pid), (send_buf, recv_buf))
    else
        while true
            try
                nbr_chnls[pid] = remotecall_fetch(
                    ClimaCommsSA.get_nbr_chnl,
                    pid,
                    ClimaComms.mypid(SACommsContext),
                )
                break
            catch e
                if e isa RemoteException && e.captured.ex isa KeyError
                    yield()
                    continue
                else
                    rethrow()
                end
            end
        end
        (send_buf, recv_buf) = take!(ClimaCommsSA.get_nbr_chnl(pid))
    end
    return ClimaComms.Neighbor{SharedArray}(pid, send_buf, recv_buf)
end

ClimaComms.get_stage(sa::SharedArray) = sdata(sa)

function ClimaComms.init(::Type{SACommsContext})
    ctx_chnl[] = remotecall_fetch(ClimaCommsSA.get_ctx_chnl, 1)
    return myid(), nprocs()
end

ClimaComms.mypid(::Type{SACommsContext}) = Distributed.myid()
ClimaComms.mypid(ctx::SACommsContext) = Distributed.myid()
ClimaComms.iamroot(CC::Type{SACommsContext}) = ClimaComms.mypid(CC) == 1
ClimaComms.iamroot(ctx::SACommsContext) = ClimaComms.mypid(ctx) == 1
ClimaComms.nprocs(::Type{SACommsContext}) = Distributed.nprocs()
ClimaComms.nprocs(::SACommsContext) = Distributed.nprocs()
ClimaComms.singlebuffered(::Type{SACommsContext}) = true

function ClimaComms.start(ctx::SACommsContext; dependencies = nothing) end

function ClimaComms.progress(ctx::SACommsContext) end

function ClimaComms.finish(ctx::SACommsContext; dependencies = nothing) end

function ClimaComms.barrier(ctx::SACommsContext)
    if nprocs() == 1
        return
    end

    barr = ctx.barrier
    if ClimaComms.iamroot(ctx)
        for pid in workers()
            while barr.worker_sense[pid] == barr.sense[1]
                yield()
            end
        end
        barr.sense[1] = !barr.sense[1]
    else
        barr.worker_sense[ClimaComms.mypid(ctx)] =
            !barr.worker_sense[ClimaComms.mypid(ctx)]
        while barr.sense[1] != barr.worker_sense[ClimaComms.mypid(ctx)]
            yield()
        end
    end
end

function ClimaComms.reduce(ctx::SACommsContext, val, op)
    if nprocs() == 1
        return val
    end

    reducer = if val isa Float64
        ctx.reducer_f64
    elseif val isa Float32
        ctx.reducer_f32
    else
        error("No reducer for type $(typeof(val))")
    end

    if ClimaComms.iamroot(ctx)
        for pid in workers()
            while reducer.worker_sense[pid] == reducer.sense[1]
                yield()
            end
            val = op(val, reducer.vals[pid])
        end
        reducer.sense[1] = !reducer.sense[1]
        return val
    else
        reducer.vals[ClimaComms.mypid(ctx)] = val
        reducer.worker_sense[ClimaComms.mypid(ctx)] =
            !reducer.worker_sense[ClimaComms.mypid(ctx)]
        while reducer.sense[1] != reducer.worker_sense[ClimaComms.mypid(ctx)]
            yield()
        end
    end
end

ClimaComms.abort(ctx::SACommsContext, status::Int) = exit(status)

end # module
