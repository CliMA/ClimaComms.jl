# Parallelize f across GPU threads, with optional coarsening. When a kernel
# configuration requires either too many blocks or too many threads in each
# block, add more coarsening until the kernel configuration is valid. Maximize
# memory coalescing for coarsened threads using grid-stride loops:
# developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops
ClimaComms.threaded_on_device(
    f::F,
    device::CUDADevice,
    itr,
    shmem_itr;
    coarsen,
    shmem_coarsen,
    block_size,
    debug,
    _...,
) where {F} =
    if isnothing(shmem_itr)
        cuda_threaded(f, itr, coarsen, block_size, debug)
    else
        threadable_shmem_itr = ClimaComms.threadable(device, shmem_itr)
        if isempty(threadable_shmem_itr)
            f_without_shmem = Base.Fix2(f, shmem_itr)
            cuda_threaded(f_without_shmem, itr, coarsen, block_size, debug)
        else
            cuda_threaded(
                f,
                itr,
                ClimaComms.shareable(threadable_shmem_itr, device),
                coarsen,
                shmem_coarsen,
                block_size,
                debug,
            )
        end
    end

# Conversion of Int64 to Int32, when the value doesn't exceed the largest Int32.
compact_int(i) = i <= typemax(Int32) ? Int32(i) : i

# Number of threads in each warp of kernel being executed.
threads_in_warp() = CUDA.warpsize()

# Number of blocks in kernel being executed and index of calling thread's block.
blocks_in_kernel() = CUDA.gridDim().x
block_idx() = CUDA.blockIdx().x

# Number of threads in each block of kernel being executed and index of calling
# thread within its block.
threads_in_block() = CUDA.blockDim().x
thread_idx_in_block() = CUDA.threadIdx().x

# Total number of threads in kernel being executed and index of calling thread.
threads_in_kernel() = threads_in_block() * blocks_in_kernel()
thread_idx_in_kernel() =
    (block_idx() - Int32(1)) * threads_in_block() + thread_idx_in_block()

# CUDA's internal representation of the GPU used for this kernel.
kernel_device(kernel) = CUDA.device(kernel.fun.mod.ctx)

# Number of threads in each warp of the GPU used for this kernel.
warp_size(kernel) =
    CUDA.attribute(kernel_device(kernel), CUDA.DEVICE_ATTRIBUTE_WARP_SIZE)

# Maximum number of blocks that can fit on the GPU used for this kernel.
grid_size_limit(kernel) =
    CUDA.attribute(kernel_device(kernel), CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)

# Either the maximum number of threads that can fit in one block of this kernel
# (cuOccupancyMaxPotentialBlockSize), or a user-specified limit. If the kernel
# has sufficiently many blocks, the default limit will maximize GPU occupancy.
block_size_limit(::Nothing, kernel) =
    CUDA.launch_configuration(kernel.fun).threads
block_size_limit(max_threads_in_block, _) = max_threads_in_block

# Compiled PTX code for a GPU kernel, with all function calls inlined.
compiled_kernel(kernel_function::F) where {F} =
    CUDA.@cuda always_inline=true launch=false kernel_function()

kernel_config_limits(kernel, block_size) =
    (grid_size_limit(kernel), block_size_limit(block_size, kernel))

print_debug_info(::Nothing, _, _, _) = nothing
print_debug_info(::Val{:warn}, kernel, _, _) =
    CUDA.@device_code_warntype kernel()
print_debug_info(::Val{:llvm}, kernel, _, _) = CUDA.@device_code_llvm kernel()
print_debug_info(::Val{:ptx}, kernel, _, _) = CUDA.@device_code_ptx kernel()
print_debug_info(::Val{:sass}, kernel, _, _) = CUDA.@device_code_sass kernel()
print_debug_info(::Val{:stats}, kernel, n_blocks, threads_in_block) =
    @info """Kernel Statistics:
             $(4 * CUDA.registers(kernel)) bytes of register memory,
             $(CUDA.memory(kernel).local) bytes of register spillover,
             $(CUDA.memory(kernel).shared) bytes of shared memory,
             $n_blocks blocks,
             $threads_in_block threads in each block,
             $(CUDA.maxthreads(kernel)) maximum threads per block"""
# TODO: Add relevant statistics about compiled code, e.g., the number of
# conditional branches, or the number of double precision operations.

# TODO: Modify loops to avoid using StepRange on GPUs:
# https://cuda.juliagpu.org/stable/tutorials/performance/#Avoiding-StepRange

function cuda_threaded(f::F, itr, coarsen, block_size, debug) where {F}
    n_items = compact_int(length(itr))
    is_coarsened = coarsen isa Integer

    if is_coarsened
        kernel = compiled_kernel() do
            for item_index in thread_idx_in_kernel():threads_in_kernel():n_items
                @inbounds f(itr[item_index])
            end
        end
    else
        kernel = compiled_kernel() do
            item_index = thread_idx_in_kernel()
            item_index <= n_items && @inbounds f(itr[item_index])
            nothing
        end
    end

    max_blocks, max_threads_in_block = kernel_config_limits(kernel, block_size)
    max_threads_in_kernel = max_blocks * max_threads_in_block
    (!is_coarsened && n_items > max_threads_in_kernel) &&
        return cuda_threaded(f, itr, 1, block_size, debug)

    min_items_in_thread = is_coarsened ? coarsen : 1
    max_required_threads_in_kernel = cld(n_items, min_items_in_thread)
    items_in_thread =
        max_required_threads_in_kernel <= max_threads_in_kernel ?
        min_items_in_thread : cld(n_items, max_threads_in_kernel)
    threads_in_block = min(max_required_threads_in_kernel, max_threads_in_block)
    n_blocks = cld(n_items, items_in_thread * threads_in_block)

    print_debug_info(debug, kernel, n_blocks, threads_in_block)
    kernel(; blocks = n_blocks, threads = threads_in_block)
end

function cuda_threaded(
    f::F,
    itr,
    shmem_itr,
    coarsen,
    shmem_coarsen,
    block_size,
    debug,
) where {F}
    n_items = compact_int(length(itr))
    n_shmem_items = compact_int(length(shmem_itr))
    is_coarsened = coarsen isa Integer
    is_shmem_coarsened = shmem_coarsen isa Integer

    compiled_kernel_with_shmem(shmem_itr) =
        if is_coarsened
            compiled_kernel() do
                for item_index in block_idx():blocks_in_kernel():n_items
                    @inbounds f(itr[item_index], shmem_itr)
                end
            end
        else
            compiled_kernel() do
                item_index = block_idx()
                item_index <= n_items && @inbounds f(itr[item_index], shmem_itr)
                nothing
            end
        end

    kernel = compiled_kernel_with_shmem(shmem_itr)

    max_blocks, max_threads_in_block = kernel_config_limits(kernel, block_size)
    coarsening_needed_but_missing = !is_coarsened && n_items > max_blocks
    shmem_coarsening_needed_but_missing =
        !is_shmem_coarsened && n_shmem_items > max_threads_in_block
    (coarsening_needed_but_missing || shmem_coarsening_needed_but_missing) &&
        return cuda_threaded(
            f,
            itr,
            shmem_itr,
            coarsening_needed_but_missing ? 1 : coarsen,
            shmem_coarsening_needed_but_missing ? 1 : shmem_coarsen,
            block_size,
            debug,
        )

    min_items_in_block = is_coarsened ? coarsen : 1
    max_required_blocks = cld(n_items, min_items_in_block)
    items_in_block =
        max_required_blocks <= max_blocks ? min_items_in_block :
        cld(n_items, max_blocks)
    n_blocks = cld(n_items, items_in_block)

    min_shmem_items_in_thread = is_shmem_coarsened ? shmem_coarsen : 1
    max_required_threads_in_block =
        cld(n_shmem_items, min_shmem_items_in_thread)
    shmem_items_in_thread =
        max_required_threads_in_block <= max_threads_in_block ?
        min_shmem_items_in_thread : cld(n_shmem_items, max_threads_in_block)
    threads_in_block = cld(n_shmem_items, shmem_items_in_thread)

    # metadata = (; threads_in_warp = warp_size(kernel), threads_in_block)
    # unrollable_shmem_itr = ClimaComms.set_metadata(shmem_itr, metadata)
    # unrolled_kernel = compiled_kernel_with_shmem(unrollable_shmem_itr)
    unrolled_kernel = kernel # TODO: Use the actual unrolled kernel.

    # TODO: Check if the launch config is still optimal for the unrolled kernel.
    # TODO: Add an optimization mode that iteratively updates the launch config.

    print_debug_info(debug, unrolled_kernel, n_blocks, threads_in_block)
    CUDA.@sync unrolled_kernel(; blocks = n_blocks, threads = threads_in_block)
end
