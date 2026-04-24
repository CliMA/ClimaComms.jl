module ClimaCommsMetalExt

import Metal

import Adapt
import ClimaComms
import ClimaComms: MetalDevice

# Metal automatically manages device assignment, so this is a no-op
ClimaComms._assign_device(::MetalDevice, rank_number) = nothing

function Base.summary(io::IO, ::MetalDevice)
    dev = Metal.device()
    name = dev.name
    return "$name (Metal)"
end

ClimaComms.device_functional(::MetalDevice) = !isempty(Metal.devices())

Adapt.adapt_structure(
    to::Type{<:Metal.MtlArray}, ctx::ClimaComms.AbstractCommsContext,
) =
    ClimaComms.context(Adapt.adapt(to, ClimaComms.device(ctx)))

Adapt.adapt_structure(::Type{<:Metal.MtlArray}, ::ClimaComms.AbstractDevice) =
    ClimaComms.MetalDevice()

ClimaComms.array_type(::MetalDevice) = Metal.MtlArray
ClimaComms.free_memory(::MetalDevice) = Metal.device().currentAllocatedSize
ClimaComms.total_memory(::MetalDevice) = Metal.device().maxBufferLength
ClimaComms.allowscalar(f, ::MetalDevice, args...; kwargs...) =
    Metal.@allowscalar f(args...; kwargs...)

# Extending ClimaComms methods that operate on expressions (cannot use dispatch here)
ClimaComms.sync(f::F, ::MetalDevice, args...; kwargs...) where {F} =
    Metal.@sync f(args...; kwargs...)
ClimaComms.cuda_sync(f::F, ::MetalDevice, args...; kwargs...) where {F} =  # TODO: Rename to `device_sync` to unify `Metal` and `CUDA`
    Metal.@sync f(args...; kwargs...)
ClimaComms.time(f::F, ::MetalDevice, args...; kwargs...) where {F} =
    Metal.@time f(args...; kwargs...)
ClimaComms.elapsed(f::F, ::MetalDevice, args...; kwargs...) where {F} =
    Metal.@elapsed f(args...; kwargs...)
ClimaComms.assert(::MetalDevice, cond::C, text::T) where {C, T} =
    isnothing(text) ? (Metal.@assert cond()) : (Metal.@assert cond() text())

# The number of threads in the kernel being executed by the calling thread.
threads_in_kernel() = Metal.threads_per_grid_1d()

# The index of the calling thread, which is between 1 and threads_in_kernel().
thread_index() = Metal.thread_position_in_grid_1d()

# The maximum number of blocks that can fit on the GPU used for this kernel.
# Metal doesn't have a direct equivalent to CUDA's max grid dim, so we use a reasonable default
grid_size_limit(kernel) = 65535

# Either the first value if it is available, or the maximum number of threads
# that can fit in one block of this kernel.
# With enough blocks, the latter value will maximize the occupancy of the GPU.
block_size_limit(max_threads_in_block::Int, _) = max_threads_in_block
block_size_limit(::Val{:auto}, kernel) =
    Int(kernel.pipeline.maxTotalThreadsPerThreadgroup)

function ClimaComms.run_threaded(
    f::F, ::MetalDevice, ::Val, itr; block_size,
) where {F}
    n_items = length(itr)
    n_items > 0 || return nothing

    function call_f_from_thread()
        item_index = thread_index()
        item_index <= n_items &&
            @inbounds f(itr[firstindex(itr) + item_index - 1])
        return nothing
    end
    kernel = Metal.@metal launch = false call_f_from_thread()
    max_blocks = grid_size_limit(kernel)
    max_threads_in_block = block_size_limit(block_size, kernel)

    params = ClimaComms._compute_launch_params_simple(
        n_items, max_blocks, max_threads_in_block,
    )
    # If there are too many items, coarsen by the smallest possible amount.
    isnothing(params) &&
        return ClimaComms.run_threaded(f, MetalDevice(), 1, itr; block_size)

    Metal.@sync kernel(;
        threads = params.threads_in_block, groups = params.blocks,
    )
end

function ClimaComms.run_threaded(
    f::F, ::MetalDevice, min_items_in_thread::Int, itr; block_size,
) where {F}
    min_items_in_thread > 0 || throw(ArgumentError("`coarsen` is not positive"))
    n_items = length(itr)
    n_items > 0 || return nothing

    # Maximize memory coalescing with a "grid-stride loop"; for reference, see
    # https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops
    call_f_from_thread() =
        for item_index in thread_index():threads_in_kernel():n_items
            @inbounds f(itr[firstindex(itr) + item_index - 1])
        end
    kernel = Metal.@metal launch = false call_f_from_thread()
    max_blocks = grid_size_limit(kernel)
    max_threads_in_block = block_size_limit(block_size, kernel)

    params = ClimaComms._compute_launch_params_coarsened(
        n_items, max_blocks, max_threads_in_block, min_items_in_thread,
    )
    Metal.@sync kernel(;
        threads = params.threads_in_block, groups = params.blocks,
    )
end

end
