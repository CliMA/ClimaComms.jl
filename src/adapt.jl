import Adapt

"""
    Adapt.adapt_structure(::Type{<:AbstractArray}, context::AbstractCommsContext)

Adapt a given context to a context with a device associated with the given array type.

# Example

```julia
Adapt.adapt_structure(Array, ClimaComms.context(ClimaComms.CUDADevice())) -> ClimaComms.CPUSingleThreaded()
```

!!! note
    By default, adapting to `Array` creates a `CPUSingleThreaded` device, and
    there is currently no way to convert to a CPUMultiThreaded device.
"""
Adapt.adapt_structure(to::Type{<:AbstractArray}, ctx::AbstractCommsContext) =
    context(Adapt.adapt(to, device(ctx)))

"""
    Adapt.adapt_structure(::Type{<:AbstractArray}, device::AbstractDevice)

Adapt a given device to a device associated with the given array type.

# Example

```julia
Adapt.adapt_structure(Array, ClimaComms.CUDADevice()) -> ClimaComms.CPUSingleThreaded()
```

!!! note
    By default, adapting to `Array` creates a `CPUSingleThreaded` device, and
    there is currently no way to convert to a CPUMultiThreaded device.
"""
Adapt.adapt_structure(::Type{<:AbstractArray}, device::AbstractDevice) =
    CPUSingleThreaded()


"""
    adapt(device::AbstractDevice, x)

Adapt an object `x` to be compatible with the specified `device`.
"""
function adapt(device::AbstractDevice, x)
    return Adapt.adapt(array_type(device), x)
end

"""
    adapt(device::AbstractCommsContext, x)

Adapt an object `x` to be compatible with the specified `context`.
"""
function adapt(context::AbstractCommsContext, x)
    return Adapt.adapt(array_type(device(context)), x)
end
