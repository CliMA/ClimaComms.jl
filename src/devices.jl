import ..ClimaComms
import Adapt
import StaticArrays

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
    ClimaComms.device_functional(device)

Return true when the `device` is correctly set up.
"""
function device_functional end

device_functional(::CPUSingleThreaded) = true
device_functional(::CPUMultiThreaded) = true

function device_type()
    env_var = get(ENV, "CLIMACOMMS_DEVICE", "CPU")
    if env_var == "CPU"
        return Threads.nthreads() > 1 ? :CPUMultiThreaded : :CPUSingleThreaded
    elseif env_var == "CPUSingleThreaded"
        return :CPUSingleThreaded
    elseif env_var == "CPUMultiThreaded"
        return :CPUMultiThreaded
    elseif env_var == "CUDA"
        return :CUDADevice
    else
        error("Invalid CLIMACOMMS_DEVICE: $env_var")
    end
end

"""
    ClimaComms.device()

Determine the device to use depending on the `CLIMACOMMS_DEVICE` environment variable.

Allowed values:
- `CPU`, single-threaded or multi-threaded depending on the number of threads;
- `CPUSingleThreaded`,
- `CPUMultiThreaded`,
- `CUDA`.

The default is `CPU`.
"""
function device()
    target_device = device_type()
    if target_device == :CUDADevice && !cuda_ext_is_loaded()
        error(
            "Loading CUDA.jl is required to use CUDADevice. You might want to call ClimaComms.@import_required_backends",
        )
    end
    DeviceConstructor = getproperty(ClimaComms, target_device)
    return DeviceConstructor()
end

Base.summary(io::IO, device::AbstractDevice) = string(device_type())

"""
    ClimaComms.array_type(::AbstractDevice)

The base array type used by the specified device (currently `Array` or `CuArray`).
"""
array_type(::AbstractCPUDevice) = Array

"""
Internal function that can be used to assign a device to a process.

Currently used to assign CUDADevices to MPI ranks.
"""
_assign_device(device, id) = nothing

"""
    ClimaComms.free_memory(device)

Bytes of memory that are currently available for allocation on the `device`.
"""
free_memory(::AbstractCPUDevice) = Sys.free_memory()

"""
    ClimaComms.total_memory(device)

Bytes of memory that are theoretically available for allocation on the `device`.
"""
total_memory(::AbstractCPUDevice) = Sys.total_memory()

"""
    @time f(args...; kwargs...)

Device-flexible `@time`:

Calls
```julia
@time f(args...; kwargs...)
```
for CPU devices and
```julia
CUDA.@time f(args...; kwargs...)
```
for CUDA devices.
"""
function time(f::F, device::AbstractCPUDevice, args...; kwargs...) where {F}
    Base.@time begin
        f(args...; kwargs...)
    end
end

"""
    elapsed(f::F, device::AbstractDevice, args...; kwargs...)

Device-flexible `elapsed`.

Calls
```julia
@elapsed f(args...; kwargs...)
```
for CPU devices and
```julia
CUDA.@elapsed f(args...; kwargs...)
```
for CUDA devices.
"""
function elapsed(f::F, device::AbstractCPUDevice, args...; kwargs...) where {F}
    Base.@elapsed begin
        f(args...; kwargs...)
    end
end

"""
    sync(f, ::AbstractDevice, args...; kwargs...)

Device-flexible function that calls `@sync`.

Calls
```julia
@sync f(args...; kwargs...)
```
for CPU devices and
```julia
CUDA.@sync f(args...; kwargs...)
```
for CUDA devices.

An example use-case of this might be:
```julia
BenchmarkTools.@benchmark begin
    if ClimaComms.device() isa ClimaComms.CUDADevice
        CUDA.@sync begin
            launch_cuda_kernels_or_spawn_tasks!(...)
        end
    elseif ClimaComms.device() isa ClimaComms.CPUMultiThreading
        Base.@sync begin
            launch_cuda_kernels_or_spawn_tasks!(...)
        end
    end
end
```

If the CPU version of the above example does not leverage
spawned tasks (which require using `Base.sync` or `Threads.wait`
to synchronize), then you may want to simply use [`cuda_sync`](@ref).
"""
function sync(f::F, ::AbstractCPUDevice, args...; kwargs...) where {F}
    Base.@sync begin
        f(args...; kwargs...)
    end
end

"""
    cuda_sync(f, ::AbstractDevice, args...; kwargs...)

Device-flexible function that (may) call `CUDA.@sync`.

Calls
```julia
f(args...; kwargs...)
```
for CPU devices and
```julia
CUDA.@sync f(args...; kwargs...)
```
for CUDA devices.
"""
function cuda_sync(f::F, ::AbstractCPUDevice, args...; kwargs...) where {F}
    f(args...; kwargs...)
end

"""
    allowscalar(f, ::AbstractDevice, args...; kwargs...)

Device-flexible version of `CUDA.@allowscalar`.

Lowers to
```julia
f(args...)
```
for CPU devices and
```julia
CUDA.@allowscalar f(args...)
```
for CUDA devices.

This is usefully written with closures via
```julia
allowscalar(device) do
    f()
end
```
"""
allowscalar(f, ::AbstractCPUDevice, args...; kwargs...) = f(args...; kwargs...)

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
    __CC__ = ClimaComms
    return :($__CC__.time(() -> $(esc(expr)), $(esc(device))))
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

An example use-case of this might be:
```julia
BenchmarkTools.@benchmark begin
    if ClimaComms.device() isa ClimaComms.CUDADevice
        CUDA.@sync begin
            launch_cuda_kernels_or_spawn_tasks!(...)
        end
    elseif ClimaComms.device() isa ClimaComms.CPUMultiThreading
        Base.@sync begin
            launch_cuda_kernels_or_spawn_tasks!(...)
        end
    end
end
```

If the CPU version of the above example does not leverage
spawned tasks (which require using `Base.sync` or `Threads.wait`
to synchronize), then you may want to simply use [`@cuda_sync`](@ref).
"""
macro sync(device, expr)
    __CC__ = ClimaComms
    return :($__CC__.sync(() -> $(esc(expr)), $(esc(device))))
end

"""
    @cuda_sync device expr

Device-flexible `CUDA.@sync`.

Lowers to
```julia
expr
```
for CPU devices and
```julia
CUDA.@sync expr
```
for CUDA devices.
"""
macro cuda_sync(device, expr)
    __CC__ = ClimaComms
    return :($__CC__.cuda_sync(() -> $(esc(expr)), $(esc(device))))
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
    __CC__ = ClimaComms
    return :($__CC__.elapsed(() -> $(esc(expr)), $(esc(device))))
end

"""
    @assert device cond [text]

Device-flexible `@assert`.

Lowers to
```julia
@assert cond [text]
```
for CPU devices and
```julia
CUDA.@cuassert cond [text]
```
for CUDA devices.
"""
macro assert(device, cond, text = nothing)
    text_func = isnothing(text) ? nothing : :(() -> $(esc(text)))
    return :($assert($(esc(device)), () -> $(esc(cond)), $text_func))
end
assert(::AbstractCPUDevice, cond::C, text::T) where {C, T} =
    isnothing(text) ? (Base.@assert cond()) : (Base.@assert cond() text())

include("threaded.jl")
include("threadable.jl")
include("shareable.jl")
include("cpu_threaded.jl")
include("cpu_shmem.jl")
