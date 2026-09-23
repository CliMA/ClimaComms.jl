module ClimaCommsCUDAExt

import CUDA

import Adapt
import ClimaComms
import ClimaComms: CUDADevice

function ClimaComms._assign_device(::CUDADevice, rank_number)
    CUDA.device!(rank_number % CUDA.ndevices())
    return nothing
end

function ClimaComms.device_synchronize(::CUDADevice)
    CUDA.device_synchronize()
    return nothing
end

function Base.summary(io::IO, ::CUDADevice)
    dev = CUDA.device()
    name = CUDA.name(dev)
    uuid = CUDA.uuid(dev)
    print(io, "$name ($uuid)")
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
ClimaComms.free_memory(::CUDADevice) = CUDA.free_memory()
ClimaComms.total_memory(::CUDADevice) = CUDA.total_memory()
ClimaComms.allowscalar(f, ::CUDADevice, args...; kwargs...) =
    CUDA.@allowscalar f(args...; kwargs...)

# CUDA methods of the functional forms underlying the device-flexible
# macros (@sync, @time, etc.); macros cannot dispatch on the device type,
# so they call these functions.
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

ClimaComms.compile_kernel(::CUDADevice, f::F) where {F} =
    CUDA.@cuda always_inline = true launch = false f()
ClimaComms.launch_kernel(::CUDADevice, kernel, blocks, threads_in_block) =
    kernel(; blocks, threads = threads_in_block)
ClimaComms.max_blocks_in_kernel(::CUDADevice, kernel) = CUDA.attribute(
    CUDA.device(kernel.fun.mod.ctx),
    CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
)
ClimaComms.max_threads_in_block(::CUDADevice, kernel, block_size::Int, _) =
    block_size
# cuOccupancyMaxPotentialBlockSize; with enough blocks, this value maximizes
# the occupancy of the GPU.
ClimaComms.max_threads_in_block(::CUDADevice, kernel, ::Val{:auto}, _) =
    CUDA.launch_configuration(kernel.fun).threads
ClimaComms.thread_index_in_kernel(::CUDADevice) =
    (Int(CUDA.blockIdx().x) - 1) * Int(CUDA.blockDim().x) +
    Int(CUDA.threadIdx().x)
ClimaComms.threads_in_kernel(::CUDADevice) =
    Int(CUDA.blockDim().x) * Int(CUDA.gridDim().x)

end
