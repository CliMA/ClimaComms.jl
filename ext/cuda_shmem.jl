# Number of threads in each warp of kernel being executed.
threads_in_warp() = CUDA.warpsize()

ClimaComms.synchronize_shmem!(::CUDADevice) = CUDA.sync_threads()

ClimaComms.shmem_array_on_device(::CUDADevice, ::Type{T}, dims...) where {T} =
    StaticArrays.SizedArray{Tuple{dims...}}(CUDA.CuStaticSharedArray(T, dims))

function ClimaComms.shmem_map_on_device!(
    f::F,
    ::CUDADevice,
    dest_array,
    src_array,
) where {F}
    length(dest_array) != length(src_array) &&
        throw(ArgumentError("Destination and source must have the same length"))
    isempty(src_array) && return nothing

    n_indices = length(src_array)
    for map_index in thread_idx_in_block():threads_in_block():n_indices
        @inbounds dest_array[map_index] = f(src_array[map_index])
    end
    n_indices > threads_in_warp() && ClimaComms.auto_synchronize!(src_array)
    return nothing
end

function ClimaComms.shmem_reduce_on_device!(
    op::O,
    device::CUDADevice,
    array;
    init...,
) where {O}
    @assert keys(init) in ((), (:init,))
    if isempty(array)
        isempty(init) &&
            throw(ArgumentError("Reduction of empty array requires init value"))
        return values(init)[1]
    end

    n_indices = length(array)
    warp_size = threads_in_warp()
    block_size = threads_in_block()
    reduction_index = thread_idx_in_block()

    if n_indices <= block_size
        n_remaining_inputs = n_indices
    else
        @inbounds reduced_value = array[reduction_index]
        for input_index in (reduction_index + block_size):block_size:n_indices
            @inbounds reduced_value = op(reduced_value, array[input_index])
        end
        @inbounds array[reduction_index] = reduced_value
        block_size > warp_size && ClimaComms.synchronize_shmem!(device)
        n_remaining_inputs = block_size
    end

    while n_remaining_inputs > 1
        n_reductions = n_remaining_inputs >> 1 # >>1 is a bitwise version of ÷2
        n_remaining_inputs -= n_reductions
        if reduction_index <= n_reductions
            second_input_index = reduction_index + n_remaining_inputs
            @inbounds array[reduction_index] =
                op(array[reduction_index], array[second_input_index])
        end
        n_reductions > warp_size && ClimaComms.synchronize_shmem!(device)
    end

    ClimaComms.auto_synchronize!(array)
    return @inbounds isempty(init) ? array[1] : op(values(init)[1], array[1])
end

ClimaComms.unique_shmem_thread_on_device(f::F, ::CUDADevice) where {F} =
    thread_idx_in_block() == 1 && f()

# TODO: Implement unrolled versions of shmem_map! and shmem_reduce!, which will
# require storing the warp_size and block_size as type parameters of the array.
