using CUDA
using KernelAbstractions
using StaticArrays

@enum RelayBufferKind begin
    SingleRelayBuffer
    DoubleRelayBuffer
end

Base.similar(::Type{A}, ::Type{FT}, dims...) where {A <: Array, FT} =
    similar(Array{FT}, dims...)
Base.similar(::Type{A}, ::Type{FT}, dims...) where {A <: CuArray, FT} =
    similar(CuArray{FT}, dims...)

device(::Union{Array, SArray, MArray}) = CPU()
device(::CUDA.CuArray) = CUDADevice()

#=
    RelayBuffer{T}(::Type{A}, kind, dims...; pinned = true)

A RelayBuffer abstracts storage for communication; it is used for
staging data and for transfers. When running on:

  - CPU -- a single buffer is used for staging and transfers can be
    initiated directly to/from it.
  - GPU -- either:
    - The transfer infrastructure is CUDA-aware and a single buffer
      can be used, just as with the CPU, or
    - The transfer infrastructure is not CUDA-aware: a double buffer
      must be used with the staging buffer on the device and the
      transfer buffer on the host.

# Arguments
- `T`: element type
- `A::Type`: what kind of array to allocate for `stage`
- `kind::RelayBufferKind`: either `SingleRelayBuffer` or
  `DoubleRelayBuffer`
- `dims...`: dimensions of the array
=#
struct RelayBuffer{T, A, Buff}
    stage::A
    transfer::Buff # Union{Nothing,Buff}

    function RelayBuffer{T}(::Type{A}, kind, dims...) where {T, A}
        stage = similar(A, T, dims...)

        if kind == SingleRelayBuffer
            transfer = nothing
        elseif kind == DoubleRelayBuffer
            transfer = zeros(T, dims...)
        else
            error("Unknown RelayBufferkind $kind")
        end

        return new{T, typeof(stage), typeof(transfer)}(stage, transfer)
    end
end

function get_stage(buf::RelayBuffer)
    return buf.stage
end

function get_transfer(buf::RelayBuffer)
    if buf.transfer === nothing
        return buf.stage
    else
        return buf.transfer
    end
end

function prepare_transfer!(
    buf::RelayBuffer;
    dependencies = nothing,
    progress = yield,
)
    if buf.transfer === nothing || length(buf.transfer) == 0
        return MultiEvent(dependencies)
    end

    event = async_copy!(
        device(buf.stage),
        buf.transfer,
        buf.stage;
        dependencies = dependencies,
        progress = progress,
    )

    return event
end

function prepare_stage!(
    buf::RelayBuffer;
    dependencies = nothing,
    progress = yield,
)
    if buf.transfer === nothing || length(buf.stage) == 0
        return MultiEvent(dependencies)
    end

    event = async_copy!(
        device(buf.stage),
        buf.stage,
        buf.transfer;
        dependencies = dependencies,
        progress = progress,
    )

    return event
end
