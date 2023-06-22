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
            Threads.nthreads() > 1 ? CPUMultiThreaded() : CPUSingleThreaded()
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
macro threaded(device, expr)
    return esc(quote
        let
            if $device isa ClimaComms.CPUMultiThreaded
                Threads.@threads $(expr)
            else
                @assert $device isa ClimaComms.CPUSingleThreaded
                $(expr)
            end
        end
    end)
end
