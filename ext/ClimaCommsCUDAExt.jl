module ClimaCommsCUDAExt

import CUDA

import ClimaComms
import ClimaComms: CUDADevice
import ClimaComms: AbstractCommsContext

function ClimaComms._assign_device(::CUDADevice, rank_number)
    CUDA.device!(rank_number % CUDA.ndevices())
    return nothing
end

function ClimaComms.device_functional(::CUDADevice)
    return CUDA.functional()
end

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

function Base.summary(io::IO, ctx::AbstractCommsContext, device::CUDADevice)
    println(io, "context: $ctx")
    println(io, "device: $device")
    println(io, "CUDA.functional() = $(CUDA.functional())")
    println(io, "CUDA.ndevices() = $(CUDA.ndevices())")
    println(io, "length(CUDA.devices()) = $(length(CUDA.devices()))")
end

end
