# we previously used CUDA as a variable
import CUDA


"""
    AbstractDevice

The base type for a device.
"""
abstract type AbstractDevice end

"""
    AbstractCPUDevice()

Abstract device type for single-threaded and multi-threaded CPU runs.
"""
abstract type AbstractCPUDevice <: AbstractDevice end


"""
    CPUSingleThreaded()

Use the CPU with single thread.
"""
struct CPUSingleThreaded <: AbstractCPUDevice end

"""
    CPUMultiThreaded()

Use the CPU with multiple thread.
"""
struct CPUMultiThreaded <: AbstractCPUDevice end

"""
    CUDADevice()

Use NVIDIA GPU accelarator
"""
struct CUDADevice <: AbstractDevice end

"""
    ClimaComms.device()

Automatically determine the appropriate device to use, returning one of
 - [`AbstractCPUDevice()`](@ref)
 - [`CUDADevice()`](@ref)

By default, it will check if a functional CUDA installation exists, using CUDA if possible.

Behavior can be overridden by setting the `CLIMACOMMS_DEVICE` environment variable to either `CPU` or `CUDA`.
"""
function device()
    env_var = get(ENV, "CLIMACOMMS_DEVICE", nothing)
    if !isnothing(env_var)
        if env_var == "CPU"
            return Threads.nthreads() > 1 ? CPUMultiThreaded() :
                   CPUSingleThreaded()
        elseif env_var == "CPUSingleThreaded"
            return CPUSingleThreaded()
        elseif env_var == "CPUMultiThreaded"
            return CPUMultiThreaded()
        elseif env_var == "CUDA"
            return CUDADevice()
        else
            error("Invalid CLIMACOMMS_DEVICE: $env_var")
        end
    end
    if CUDA.functional()
        return CUDADevice()
    else
        return Threads.nthreads() == 1 ? CPUSingleThreaded() :
               CPUMultiThreaded()
    end
end

"""
    ClimaComms.array_type(::AbstractDevice)

The base array type used by the specified device (currently `Array` or `CuArray`).
"""
array_type(::AbstractCPUDevice) = Array
array_type(::CUDADevice) = CUDA.CuArray


"""
    @threaded device for ... end

A threading macro that uses Julia native threading if the
device is a `CPUMultiThreaded` type, otherwise return
the original expression without `Threads.@threads`. This is
done to avoid overhead from `Threads.@threads`, and the device
is used (instead of checking `Threads.nthreads() == 1`) so
that this is statically inferred.

## References

 - https://discourse.julialang.org/t/threads-threads-with-one-thread-how-to-remove-the-overhead/58435
 - https://discourse.julialang.org/t/overhead-of-threads-threads/53964
"""
macro threaded(device, loop)
    quote
        if $(esc(device)) isa CPUMultiThreaded
            Threads.@threads $(Expr(
                loop.head,
                Expr(loop.args[1].head, esc.(loop.args[1].args)...),
                esc(loop.args[2]),
            ))
        else
            @assert $(esc(device)) isa AbstractDevice
            $(esc(loop))
        end
    end
end

"""
    @time device expr

Device-flexible `@time`.

Lowers to
```julia
@time expr
```
for CPU devices and
```julia
CUDA.@time expr
```
for CUDA devices.
"""
macro time(device, expr)
    return quote
        if $(esc(device)) isa CUDADevice
            CUDA.@time $(esc(expr))
        else
            @assert $(esc(device)) isa AbstractDevice
            Base.@time $(esc(expr))
        end
    end
end

"""
    @elapsed device expr

Device-flexible `@elapsed`.

Lowers to
```julia
@elapsed expr
```
for CPU devices and
```julia
CUDA.@elapsed expr
```
for CUDA devices.
"""
macro elapsed(device, expr)
    return quote
        if $(esc(device)) isa CUDADevice
            CUDA.@elapsed $(esc(expr))
        else
            @assert $(esc(device)) isa AbstractDevice
            Base.@elapsed $(esc(expr))
        end
    end
end

"""
    @sync device expr

Device-flexible `@sync`.

Lowers to
```julia
@sync expr
```
for CPU devices and
```julia
CUDA.@sync expr
```
for CUDA devices.
"""
macro sync(device, expr)
    # https://github.com/JuliaLang/julia/issues/28979#issuecomment-1756145207
    return esc(quote
        if $(device) isa $CUDADevice
            $CUDA.@sync begin
                $(expr)
            end
        else
            @assert $(device) isa $AbstractDevice
            $Base.@sync begin
                $(expr)
            end
        end
    end)
end
