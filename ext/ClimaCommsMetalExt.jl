module ClimaCommsMetalExt

import Metal

import Adapt
import ClimaComms
import ClimaComms: MetalDevice

"""
    ClimaComms._assign_device(::MetalDevice, rank_number)

Assign a Metal device to an MPI rank.
"""
function ClimaComms._assign_device(::MetalDevice, rank_number)
    # Metal automatically manages device assignment, so this is a no-op
    return nothing
end

"""
    Base.summary(io::IO, ::MetalDevice)

Print a summary of the Metal device.
"""
function Base.summary(io::IO, ::MetalDevice)
    # Get the default Metal device
    dev = Metal.devices()[1]
    name = dev.name
    return "$name (Metal)"
end

"""
    ClimaComms.device_functional(::MetalDevice)

Check if the Metal device is functional.
"""
function ClimaComms.device_functional(::MetalDevice)
    return !isempty(Metal.devices())
end

"""
    Adapt.adapt_structure(to::Type{<:Metal.MtlArray}, ctx::ClimaComms.AbstractCommsContext)

Adapt a communication context to use Metal arrays.
"""
function Adapt.adapt_structure(
    to::Type{<:Metal.MtlArray},
    ctx::ClimaComms.AbstractCommsContext,
)
    return ClimaComms.context(Adapt.adapt(to, ClimaComms.device(ctx)))
end

"""
    Adapt.adapt_structure(::Type{<:Metal.MtlArray}, device::ClimaComms.AbstractDevice)

Adapt a device to use Metal arrays.
"""
Adapt.adapt_structure(
    ::Type{<:Metal.MtlArray},
    device::ClimaComms.AbstractDevice,
) = ClimaComms.MetalDevice()

"""
    ClimaComms.array_type(::MetalDevice)

Get the array type for Metal device.
"""
ClimaComms.array_type(::MetalDevice) = Metal.MtlArray

"""
    ClimaComms.allowscalar(f, ::MetalDevice, args...; kwargs...)

Device-flexible version of scalar operations for Metal.
"""
ClimaComms.allowscalar(f, ::MetalDevice, args...; kwargs...) =
    Metal.@allowscalar f(args...; kwargs...)

# Extending ClimaComms methods that operate on expressions
ClimaComms.sync(f::F, ::MetalDevice, args...; kwargs...) where {F} =
    Metal.@sync f(args...; kwargs...)
ClimaComms.cuda_sync(f::F, ::MetalDevice, args...; kwargs...) where {F} =
    Metal.@sync f(args...; kwargs...)
ClimaComms.time(f::F, ::MetalDevice, args...; kwargs...) where {F} =
    Metal.@time f(args...; kwargs...)
ClimaComms.elapsed(f::F, ::MetalDevice, args...; kwargs...) where {F} =
    Metal.@elapsed f(args...; kwargs...)
ClimaComms.assert(::MetalDevice, cond::C, text::T) where {C, T} =
    isnothing(text) ? (Metal.@assert cond()) : (Metal.@assert cond() text())

end
