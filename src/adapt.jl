import Adapt

"""
    Adapt.adapt_structure(to::Type{<:AbstractArray}, ctx::AbstractCommsContext)

Adapt a given context to a context whose device is associated with the
given array type.

# Examples
```julia
julia> Adapt.adapt(Array, ClimaComms.context(ClimaComms.CUDADevice()))
ClimaComms.SingletonCommsContext{ClimaComms.CPUSingleThreaded}(ClimaComms.CPUSingleThreaded())
```

!!! note
    Adapting to `Array` always creates a [`CPUSingleThreaded`](@ref)
    device; there is currently no way to convert to a
    [`CPUMultiThreaded`](@ref) device.
"""
Adapt.adapt_structure(to::Type{<:AbstractArray}, ctx::AbstractCommsContext) =
    context(Adapt.adapt(to, device(ctx)))

"""
    Adapt.adapt_structure(to::Type{<:AbstractArray}, device::AbstractDevice)

Adapt a given device to the device associated with the given array type.

# Examples
```julia
julia> Adapt.adapt(Array, ClimaComms.CUDADevice())
ClimaComms.CPUSingleThreaded()
```

!!! note
    Adapting to `Array` always creates a [`CPUSingleThreaded`](@ref)
    device; there is currently no way to convert to a
    [`CPUMultiThreaded`](@ref) device.
"""
Adapt.adapt_structure(::Type{<:AbstractArray}, device::AbstractDevice) =
    CPUSingleThreaded()
