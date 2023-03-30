# we previously used CUDA as a variable
import CUDA as CUDA_jl


"""
    AbstractDevice

The base type for a device.
"""
abstract type AbstractDevice end

"""
    CPUDevice()

Use the CPU.
"""
struct CPUDevice <: AbstractDevice end

Base.@deprecate_binding CPU CPUDevice false

"""
    CUDADevice()

Use NVIDIA GPU accelarator
"""
struct CUDADevice <: AbstractDevice end

Base.@deprecate_binding CUDA CUDADevice false

"""
    ClimaComms.device()

Automatically determine the appropriate device to use, returning one of
 - [`CPUDevice()`](@ref)
 - [`CUDADevice()`](@ref)

By default, it will check if a functional CUDA installation exists, using CUDA if possible.

Behavior can be overridden by setting the `CLIMACOMMS_DEVICE` environment variable to either `CPU` or `CUDA`.
"""
function device()
    env_var = get(ENV, "CLIMACOMMS_DEVICE", nothing)
    if !isnothing(env_var)
        if env_var == "CPU"
            return CPUDevice()
        elseif env_var == "CUDA"
            return CUDADevice()
        else
            error("Invalid CLIMACOMMS_DEVICE: $env_var")
        end
    end
    if CUDA_jl.functional()
        return CUDADevice()
    else
        return CPUDevice()
    end
end

"""
    ClimaComms.device(object)

Return the appropriate device for the specified object.
"""
device(::Array) = CPUDevice()
device(::CUDA_jl.CuArray) = CUDADevice()
device(arr::SubArray) = device(parent(arr))

"""
    ClimaComms.array_type(::AbstractDevice)

The base array type used by the specified device (currently `Array` or `CuArray`).
"""
array_type(::CPUDevice) = Array
array_type(::CUDADevice) = CUDA_jl.CuArray
