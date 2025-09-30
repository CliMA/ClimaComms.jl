needs_metadata_to_unroll_shmem_loops(::AbstractCPUDevice) = false

sync_shmem_threads!(::AbstractCPUDevice) = nothing

shmem_thread_indices(::AbstractCPUDevice, n_items) = Base.OneTo(n_items)

unrolled_shmem_thread_indices(
    ::AbstractCPUDevice,
    ::Val{n_items},
) where {n_items} = StaticArrays.SOneTo(n_items)

unwrapped_shmem_array(::AbstractCPUDevice, ::Type{T}, dims) where {T} =
    StaticArrays.MArray{Tuple{dims...}, T}(undef)

unique_shmem_thread(f::F, ::AbstractCPUDevice) where {F} = f()

reduce_on_device!(op::O, ::AbstractCPUDevice, itr; init...) where {O} =
    Base.reduce(op, unwrap(itr); init...)

reduce_on_device!(op::O, ::AbstractCPUDevice, dest, itr; init...) where {O} =
    dest[] = Base.reduce(op, unwrap(itr); init...)

mapreduce_on_device!(
    f::F,
    op::O,
    ::AbstractCPUDevice,
    itr;
    init...,
) where {F, O} = mapreduce(f, op, unwrap(itr); init...)

mapreduce_on_device!(
    f::F,
    op::O,
    ::AbstractCPUDevice,
    dest,
    itr;
    init...,
) where {F, O} = dest[] = mapreduce(f, op, unwrap(itr); init...)

reduce_in_place!(_, device::AbstractCPUDevice, _) =
    error("reduce_in_place! is not supported on $device devices")

unrolled_reduce_in_place!(_, device::AbstractCPUDevice, _, _) =
    error("unrolled_reduce_in_place! is not supported on $device devices")
