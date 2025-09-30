sync_threads_in_block() = CUDA.sync_threads()

ClimaComms.needs_metadata_to_unroll_shmem_loops(::CUDADevice) = true

ClimaComms.sync_shmem_threads!(::CUDADevice) = sync_threads_in_block()

ClimaComms.shmem_thread_indices(::CUDADevice, n_items) =
    thread_idx_in_block():threads_in_block():n_items

@generated function ClimaComms.unrolled_shmem_thread_indices(
    ::CUDADevice,
    ::Val{n_items},
    ::Val{metadata},
) where {n_items, metadata}
    block_size = metadata.threads_in_block
    thread_index_expr = :(thread_index = thread_idx_in_block())
    if n_items <= block_size
        indices_expr = :((thread_index,))
    elseif n_items % block_size != 0
        indices_expr = :(thread_index:($block_size):($n_items))
    else
        index_func_expr = :(i -> (i - Int32(1)) * $block_size + thread_index)
        indices_expr = :(ntuple($index_func_expr, Val($(n_items ÷ block_size))))
    end
    return :($thread_index_expr; $indices_expr)
end

ClimaComms.unwrapped_shmem_array(::CUDADevice, ::Type{T}, dims) where {T} =
    StaticArrays.SizedArray{Tuple{dims...}}(CUDA.CuStaticSharedArray(T, dims))

function ClimaComms.unique_shmem_thread(f::F, ::CUDADevice) where {F}
    thread_idx_in_block() == 1 && f()
    return nothing
end

function ClimaComms.reduce_in_place!(op::O, ::CUDADevice, itr) where {O}
    n_items = length(itr)
    warp_size = threads_in_warp()
    block_size = threads_in_block()
    thread_index = thread_idx_in_block()

    if n_items > block_size
        @inbounds reduced_value = itr[thread_index]
        input_index = thread_index + block_size
        while input_index <= n_items
            @inbounds reduced_value = op(reduced_value, itr[input_index])
            input_index += block_size
        end
        @inbounds itr[thread_index] = reduced_value
        block_size > warp_size && sync_threads_in_block()
    end

    n_remaining_inputs = min(n_items, block_size)
    while n_remaining_inputs > 1
        n_reductions = n_remaining_inputs >> Int32(1) # bitwise version of ÷ 2
        n_remaining_inputs -= n_reductions
        if thread_index <= n_reductions
            second_input_index = thread_index + n_remaining_inputs
            @inbounds itr[thread_index] =
                op(itr[thread_index], itr[second_input_index])
        end
        n_reductions > warp_size && sync_threads_in_block()
    end
end

@generated function ClimaComms.unrolled_reduce_in_place!(
    op::O,
    ::CUDADevice,
    itr,
    ::Val{n_items},
    ::Val{metadata},
) where {O, n_items, metadata}
    warp_size = metadata.threads_in_warp
    block_size = metadata.threads_in_block

    function_body_expr = quote
        thread_index = thread_idx_in_block()
    end
    sync_expr = :(sync_threads_in_block())

    if n_items > block_size
        if n_items % block_size != 0
            reduced_value_update_expr = quote
                input_index = thread_index + $block_size
                while input_index <= $n_items
                    @inbounds reduced_value =
                        op(reduced_value, itr[input_index])
                    input_index += $block_size
                end
            end
        else
            reduced_value_update_expr = quote
                Base.Cartesian.@nexprs $(n_items ÷ block_size - 1) i ->
                    begin
                        input_index = i * $block_size + thread_index
                        @inbounds reduced_value =
                            op(reduced_value, itr[input_index])
                    end
            end
        end
        initial_reduction_expr = quote
            @inbounds reduced_value = itr[thread_index]
            $reduced_value_update_expr
            @inbounds itr[thread_index] = reduced_value
        end
        push!(function_body_expr.args, initial_reduction_expr)
        block_size > warp_size && push!(function_body_expr.args, sync_expr)
    end

    n_remaining_inputs = min(n_items, block_size)
    while n_remaining_inputs > 1
        n_reductions = n_remaining_inputs ÷ 2
        n_remaining_inputs -= n_reductions
        next_reduction_expr = quote
            if thread_index <= $n_reductions
                second_input_index = thread_index + $n_remaining_inputs
                @inbounds itr[thread_index] =
                    op(itr[thread_index], itr[second_input_index])
            end
        end
        push!(function_body_expr.args, next_reduction_expr)
        n_reductions > warp_size && push!(function_body_expr.args, sync_expr)
    end

    return function_body_expr
end

# TODO: Check whether using UnrolledUtilities can reduce latency.
