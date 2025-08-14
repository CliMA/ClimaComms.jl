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
    verbose,
    _...,
) where {F} =
    if isnothing(shmem_itr)
        cuda_threaded(f, itr, coarsen, block_size, verbose)
    else
        threadable_shmem_itr = ClimaComms.threadable(device, shmem_itr)
        if isempty(threadable_shmem_itr)
            f_without_shmem = Base.Fix2(f, shmem_itr)
            cuda_threaded(f_without_shmem, itr, coarsen, block_size, verbose)
        else
            cuda_threaded(
                f,
                itr,
                threadable_shmem_itr,
                coarsen,
                shmem_coarsen,
                block_size,
                verbose,
            )
        end
    end

# Conversion of Int64 to Int32, when the value doesn't exceed the largest Int32.
compact_int(i) = i <= typemax(Int32) ? Int32(i) : i

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

# Maximum number of blocks that can fit on the GPU used for this kernel.
grid_size_limit(kernel) = CUDA.attribute(
    CUDA.device(kernel.fun.mod.ctx),
    CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
)

# Either the maximum number of threads that can fit in one block of this kernel
# (cuOccupancyMaxPotentialBlockSize), or a user-specified limit. If the kernel
# has sufficiently many blocks, the default limit will maximize GPU occupancy.
block_size_limit(::Val{:auto}, kernel) =
    CUDA.launch_configuration(kernel.fun).threads
block_size_limit(max_threads_in_block, _) = max_threads_in_block

# Compiled PTX code for a GPU kernel, with all function calls inlined.
compiled_kernel(kernel_function::F) where {F} =
    CUDA.@cuda always_inline=true launch=false kernel_function()

kernel_config_limits(kernel, block_size) =
    (grid_size_limit(kernel), block_size_limit(block_size, kernel))

# TODO: Add relevant information about compiled PTX code: conditional branches,
# double precision operations, etc.
print_kernel_info(kernel, blocks, threads_in_block) =
    @info """Kernel Statistics:
             $(4 * CUDA.registers(kernel)) bytes of register memory,
             $(CUDA.memory(kernel).local) bytes of register spillover,
             $(CUDA.memory(kernel).shared) bytes of shared memory,
             $blocks blocks,
             $threads_in_block threads in each block,
             $(CUDA.maxthreads(kernel)) maximum threads per block"""

function cuda_threaded(f::F, itr, coarsen, block_size, verbose) where {F}
    n_items = compact_int(length(itr))
    no_coarsening = coarsen isa Val

    if no_coarsening
        kernel = compiled_kernel() do
            item_index = thread_idx_in_kernel()
            if item_index <= n_items
                @inbounds f(itr[item_index])
            end
            nothing
        end
    else
        kernel = compiled_kernel() do
            for item_index in thread_idx_in_kernel():threads_in_kernel():n_items
                @inbounds f(itr[item_index])
            end
        end
    end

    max_blocks, max_threads_in_block = kernel_config_limits(kernel, block_size)
    max_threads_in_kernel = max_blocks * max_threads_in_block
    (no_coarsening && n_items > max_threads_in_kernel) &&
        return cuda_threaded(f, itr, 1, block_size, verbose)

    min_items_in_thread = no_coarsening ? 1 : coarsen
    max_required_threads_in_kernel = cld(n_items, min_items_in_thread)
    items_in_thread =
        max_required_threads_in_kernel <= max_threads_in_kernel ?
        min_items_in_thread : cld(n_items, max_threads_in_kernel)
    threads_in_block = min(max_required_threads_in_kernel, max_threads_in_block)
    blocks = cld(n_items, items_in_thread * threads_in_block)

    verbose && print_kernel_info(kernel, blocks, threads_in_block)
    kernel(; blocks, threads = threads_in_block)
end

function cuda_threaded(
    f::F,
    itr,
    shmem_itr,
    coarsen,
    shmem_coarsen,
    block_size,
    verbose,
) where {F}
    n_items = compact_int(length(itr))
    n_shmem_items = compact_int(length(shmem_itr))
    no_coarsening = coarsen isa Val
    no_shmem_coarsening = shmem_coarsen isa Val

    shmem_itr_size_val = Val(size(shmem_itr))
    subset_of_shmem_itr(shmem_items) = ClimaComms.syncable(
        ClimaComms.static_resize(shmem_items, shmem_itr_size_val),
        CUDADevice(),
    )

    if no_coarsening && no_shmem_coarsening
        kernel = compiled_kernel() do
            item_index = block_idx()
            shmem_item_index = thread_idx_in_block()
            if item_index <= n_items && shmem_item_index <= n_shmem_items
                @inbounds shmem_itr′ =
                    subset_of_shmem_itr((shmem_itr[shmem_item_index],))
                @inbounds f(itr[item_index], shmem_itr′)
            end
            nothing
        end
    elseif no_coarsening
        kernel = compiled_kernel() do
            item_index = block_idx()
            if item_index <= n_items
                shmem_item_indices =
                    thread_idx_in_block():threads_in_block():n_shmem_items
                @inbounds shmem_itr′ =
                    subset_of_shmem_itr(view(shmem_itr, shmem_item_indices))
                @inbounds f(itr[item_index], shmem_itr′)
            end
            nothing
        end
    elseif no_shmem_coarsening
        kernel = compiled_kernel() do
            shmem_item_index = thread_idx_in_block()
            if shmem_item_index <= n_shmem_items
                @inbounds shmem_itr′ =
                    subset_of_shmem_itr((shmem_itr[shmem_item_index],))
                for item_index in block_idx():blocks_in_kernel():n_items
                    @inbounds f(itr[item_index], shmem_itr′)
                end
            end
        end
    else
        kernel = compiled_kernel() do
            shmem_item_indices =
                thread_idx_in_block():threads_in_block():n_shmem_items
            @inbounds shmem_itr′ =
                subset_of_shmem_itr(view(shmem_itr, shmem_item_indices))
            for item_index in block_idx():blocks_in_kernel():n_items
                @inbounds f(itr[item_index], shmem_itr′)
            end
        end
    end

    max_blocks, max_threads_in_block = kernel_config_limits(kernel, block_size)
    coarsening_needed_but_missing = no_coarsening && n_items > max_blocks
    shmem_coarsening_needed_but_missing =
        no_shmem_coarsening && n_shmem_items > max_threads_in_block
    (coarsening_needed_but_missing || shmem_coarsening_needed_but_missing) &&
        return cuda_threaded(
            f,
            itr,
            shmem_itr,
            coarsening_needed_but_missing ? 1 : coarsen,
            shmem_coarsening_needed_but_missing ? 1 : shmem_coarsen,
            block_size,
            verbose,
        )

    min_items_in_block = no_coarsening ? 1 : coarsen
    max_required_blocks = cld(n_items, min_items_in_block)
    items_in_block =
        max_required_blocks <= max_blocks ? min_items_in_block :
        cld(n_items, max_blocks)
    blocks = cld(n_items, items_in_block)

    min_shmem_items_in_thread = no_shmem_coarsening ? 1 : shmem_coarsen
    max_required_threads_in_block =
        cld(n_shmem_items, min_shmem_items_in_thread)
    shmem_items_in_thread =
        max_required_threads_in_block <= max_threads_in_block ?
        min_shmem_items_in_thread : cld(n_shmem_items, max_threads_in_block)
    threads_in_block = cld(n_shmem_items, shmem_items_in_thread)

    verbose && print_kernel_info(kernel, blocks, threads_in_block)
    CUDA.@sync kernel(; blocks, threads = threads_in_block)
end

# TODO: Add a keyword argument for toggling CUDA.@device_code_warntype.
