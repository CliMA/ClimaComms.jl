needs_metadata_to_unroll_shmem_loops(::AbstractCPUDevice) = false

shmem_thread_indices(::AbstractCPUDevice, itr) = Base.OneTo(length(itr))

unrolled_shmem_thread_indices(::AbstractCPUDevice, ::Val{N}) where {N} =
    StaticArrays.SOneTo(N)

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
