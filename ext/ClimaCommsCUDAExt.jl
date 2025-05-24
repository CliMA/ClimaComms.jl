module ClimaCommsCUDAExt

import CUDA

import Adapt
import ClimaComms
import ClimaComms: CUDADevice, threaded
import ClimaComms: OneInterdependentItem, MultipleInterdependentItems

function ClimaComms._assign_device(::CUDADevice, rank_number)
    CUDA.device!(rank_number % CUDA.ndevices())
    return nothing
end

function Base.summary(io::IO, ::CUDADevice)
    dev = CUDA.device()
    name = CUDA.name(dev)
    uuid = CUDA.uuid(dev)
    return "$name ($uuid)"
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

ClimaComms.synchronize_gpu_threads(::CUDADevice) = CUDA.sync_threads()

ClimaComms.static_shared_memory_array(
    ::CUDADevice,
    ::Type{T},
    dims...,
) where {T} = CUDA.CuStaticSharedArray(T, dims)

# Number of blocks in kernel being executed and index of calling thread's block.
blocks_in_kernel() = CUDA.gridDim().x
block_idx_in_kernel() = CUDA.blockIdx().x

# Number of threads in each block of kernel being executed and index of calling
# thread within its block.
threads_in_block() = CUDA.blockDim().x
thread_idx_in_block() = CUDA.threadIdx().x

# Total number of threads in kernel being executed and index of calling thread.
threads_in_kernel() = CUDA.blockDim().x * CUDA.gridDim().x
thread_idx_in_kernel() =
    (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x

# Maximum number of blocks that can fit on the GPU used for this kernel.
grid_size_limit(kernel) = CUDA.attribute(
    CUDA.device(kernel.fun.mod.ctx),
    CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
)

# Either the first value if it is available, or the maximum number of threads
# that can fit in one block of this kernel (cuOccupancyMaxPotentialBlockSize).
# With enough blocks, the latter value will maximize the occupancy of the GPU.
block_size_limit(max_threads_in_block::Int, _) = max_threads_in_block
block_size_limit(::Val{:auto}, kernel) =
    CUDA.launch_configuration(kernel.fun).threads

function threaded(f::F, device::CUDADevice, ::Val, itr; block_size) where {F}
    length(itr) > 0 || return nothing
    Base.require_one_based_indexing(itr)

    function thread_function()
        itr_index = thread_idx_in_kernel()
        itr_index <= length(itr) && @inbounds f(itr[itr_index])
        return nothing
    end
    kernel = CUDA.@cuda launch=false thread_function()
    max_blocks = grid_size_limit(kernel)
    max_threads_in_block = block_size_limit(block_size, kernel)

    # If there are too many items, coarsen by the smallest possible amount.
    length(itr) <= max_blocks * max_threads_in_block ||
        return threaded(f, device, 1, itr; block_size)

    threads_in_block = min(max_threads_in_block, length(itr))
    blocks = cld(length(itr), threads_in_block)
    CUDA.@sync kernel(; blocks, threads = threads_in_block)
end

function threaded(
    f::F,
    ::CUDADevice,
    min_items_in_thread::Int,
    itr;
    block_size,
) where {F}
    min_items_in_thread > 0 ||
        throw(ArgumentError("integer `coarsen` value must be positive"))
    length(itr) > 0 || return nothing
    Base.require_one_based_indexing(itr)

    # Maximize memory coalescing with a "grid-stride loop"; for reference, see
    # https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops
    coarsened_thread_function() =
        for itr_index in thread_idx_in_kernel():threads_in_kernel():length(itr)
            @inbounds f(itr[itr_index])
        end
    kernel = CUDA.@cuda launch=false coarsened_thread_function()
    max_blocks = grid_size_limit(kernel)
    max_threads_in_block = block_size_limit(block_size, kernel)

    # If there are too many items to use the specified coarsening, increase it
    # by the smallest possible amount.
    max_required_threads = cld(length(itr), min_items_in_thread)
    items_in_thread =
        max_required_threads <= max_blocks * max_threads_in_block ?
        min_items_in_thread :
        cld(length(itr), max_blocks * max_threads_in_block)

    threads_in_block = min(max_threads_in_block, max_required_threads)
    blocks = cld(length(itr), items_in_thread * threads_in_block)
    CUDA.@sync kernel(; blocks, threads = threads_in_block)
end

function threaded(
    f::F,
    device::CUDADevice,
    ::Union{Val, NTuple{2, Val}},
    independent_itr,
    interdependent_itr;
    block_size,
) where {F}
    length(independent_itr) > 0 || return nothing
    length(interdependent_itr) > 0 || return nothing
    Base.require_one_based_indexing(independent_itr)
    Base.require_one_based_indexing(interdependent_itr)

    function two_itr_thread_function()
        block_index = block_idx_in_kernel()
        thread_index = thread_idx_in_block()
        (
            block_index <= length(independent_itr) &&
            thread_index <= length(interdependent_itr)
        ) && @inbounds f(
            independent_itr[block_index],
            OneInterdependentItem(interdependent_itr[thread_index], device),
        )
        return nothing
    end
    kernel = CUDA.@cuda launch=false two_itr_thread_function()
    max_blocks = grid_size_limit(kernel)
    max_threads_in_block = block_size_limit(block_size, kernel)

    # If there are too many items, coarsen by the smallest possible amount.
    (
        length(independent_itr) <= max_blocks &&
        length(interdependent_itr) <= max_threads_in_block
    ) || return threaded(
        f,
        device,
        (1, 1),
        independent_itr,
        interdependent_itr;
        block_size,
    )

    blocks = length(independent_itr)
    threads_in_block = length(interdependent_itr)
    CUDA.@sync kernel(; blocks, threads = threads_in_block)
end

# Use a default coarsen value of 1 for either iterator when a value is needed.
threaded(
    f::F,
    device::CUDADevice,
    min_independent_items_in_thread::Int,
    independent_itr,
    interdependent_itr;
    block_size,
) where {F} = threaded(
    f,
    device,
    (min_independent_items_in_thread, 1),
    independent_itr,
    interdependent_itr;
    block_size,
)
threaded(
    f::F,
    device::CUDADevice,
    min_items_in_thread::Tuple{Val, Int},
    independent_itr,
    interdependent_itr;
    block_size,
) where {F} = threaded(
    f,
    device,
    (1, min_items_in_thread[2]),
    independent_itr,
    interdependent_itr;
    block_size,
)

function threaded(
    f::F,
    device::CUDADevice,
    min_items_in_thread::NTuple{2, Int},
    independent_itr,
    interdependent_itr;
    block_size,
) where {F}
    (min_items_in_thread[1] > 0 && min_items_in_thread[2] > 0) ||
        throw(ArgumentError("all integer `coarsen` values must be positive"))
    length(independent_itr) > 0 || return nothing
    length(interdependent_itr) > 0 || return nothing
    Base.require_one_based_indexing(independent_itr)
    Base.require_one_based_indexing(interdependent_itr)

    # Maximize memory coalescing with a "grid-stride loop" (reference is above).
    function coarsened_two_itr_thread_function()
        independent_itr_indices =
            block_idx_in_kernel():blocks_in_kernel():length(independent_itr)
        interdependent_itr_indices =
            thread_idx_in_block():threads_in_block():length(interdependent_itr)
        for independent_itr_index in independent_itr_indices
            @inbounds f(
                independent_itr[independent_itr_index],
                MultipleInterdependentItems(
                    interdependent_itr,
                    interdependent_itr_indices,
                    device,
                ),
            )
        end
    end
    kernel = CUDA.@cuda launch=false coarsened_two_itr_thread_function()
    max_blocks = grid_size_limit(kernel)
    max_threads_in_block = block_size_limit(block_size, kernel)

    # If there are too many items to use the specified coarsening, increase it
    # by the smallest possible amount.
    max_required_blocks = cld(length(independent_itr), min_items_in_thread[1])
    max_required_threads_in_block =
        cld(length(interdependent_itr), min_items_in_thread[2])
    items_in_thread = (
        max_required_blocks <= max_blocks ? min_items_in_thread[1] :
        cld(length(independent_itr), max_blocks),
        max_required_threads_in_block <= max_threads_in_block ?
        min_items_in_thread[2] :
        cld(length(interdependent_itr), max_threads_in_block),
    )

    blocks = cld(length(independent_itr), items_in_thread[1])
    threads_in_block = cld(length(interdependent_itr), items_in_thread[2])
    CUDA.@sync kernel(; blocks, threads = threads_in_block)
end

end
