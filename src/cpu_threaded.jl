function threaded_on_device(
    f::F,
    device::AbstractCPUDevice,
    itr,
    shmem_itr;
    coarsen,
    _...,
) where {F}
    f_without_shmem = isnothing(shmem_itr) ? f : Base.Fix2(f, shmem_itr)
    cpu_threaded(f_without_shmem, device, itr, coarsen)
end

cpu_threaded(f::F, ::CPUSingleThreaded, itr, _) where {F} =
    for item in itr
        f(item)
    end

cpu_threaded(f::F, ::CPUMultiThreaded, itr, ::Nothing) where {F} =
    Threads.@threads :static for item in itr
        f(item)
    end

cpu_threaded(f::F, ::CPUMultiThreaded, itr, ::Val{:dynamic}) where {F} =
    Threads.@threads :dynamic for item in itr
        f(item)
    end

@static if VERSION >= v"1.11"
    cpu_threaded(f::F, ::CPUMultiThreaded, itr, ::Val{:greedy}) where {F} =
        Threads.@threads :greedy for item in itr
            f(item)
        end
end

function cpu_threaded(f::F, ::CPUMultiThreaded, itr, items_in_thread) where {F}
    n_items = length(itr)
    Threads.@threads :static for thread_index in 1:cld(n_items, items_in_thread)
        first_item_index = items_in_thread * (thread_index - 1) + 1
        last_item_index = min(items_in_thread * thread_index, n_items)
        for item_index in first_item_index:last_item_index
            @inbounds f(itr[item_index])
        end
    end
end
