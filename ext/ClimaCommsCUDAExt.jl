module ClimaCommsCUDAExt

import CUDA

import Adapt
import ClimaComms
import ClimaComms: CUDADevice

function ClimaComms._assign_device(::CUDADevice, rank_number)
    CUDA.device!(rank_number % CUDA.ndevices())
    return nothing
end

function ClimaComms.device_functional(::CUDADevice)
    return CUDA.functional()
end

function Adapt.adapt_structure(
    to::Type{<:CUDA.CuArray},
    ctx::ClimaComms.AbstractCommsContext,
)
    return ClimaComms.context(Adapt.adapt(to, ClimaComms.device(ctx)))
end

Adapt.adapt_structure(
    ::Type{<:CUDA.CuArray},
    device::ClimaComms.AbstractDevice,
) = ClimaComms.CUDADevice()

ClimaComms.array_type(::CUDADevice) = CUDA.CuArray
ClimaComms.allowscalar(f, ::CUDADevice, args...; kwargs...) =
    CUDA.@allowscalar f(args...; kwargs...)

# Extending ClimaComms methods that operate on expressions (cannot use dispatch here)
ClimaComms.sync(f::F, ::CUDADevice, args...; kwargs...) where {F} =
    CUDA.@sync f(args...; kwargs...)
ClimaComms.cuda_sync(f::F, ::CUDADevice, args...; kwargs...) where {F} =
    CUDA.@sync f(args...; kwargs...)
ClimaComms.time(f::F, ::CUDADevice, args...; kwargs...) where {F} =
    CUDA.@time f(args...; kwargs...)
ClimaComms.elapsed(f::F, ::CUDADevice, args...; kwargs...) where {F} =
    CUDA.@elapsed f(args...; kwargs...)
ClimaComms.assert(::CUDADevice, cond::C, text::T) where {C, T} =
    isnothing(text) ? (CUDA.@cuassert cond()) : (CUDA.@cuassert cond() text())

end
