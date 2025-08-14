synchronize_shmem!(::AbstractCPUDevice) = nothing

shmem_array_on_device(::AbstractCPUDevice, ::Type{T}, dims...) where {T} =
    StaticArrays.MArray{Tuple{dims...}, T}(undef)

shmem_map_on_device!(f::F, ::AbstractCPUDevice, arrays...) where {F} =
    map!(f, map(unwrap, arrays)...)

shmem_reduce_on_device!(op::O, ::AbstractCPUDevice, array; init...) where {O} =
    Base.reduce(op, unwrap(array); init...)

shmem_mapreduce_on_device!(
    f::F,
    op::O,
    ::AbstractCPUDevice,
    cache,
    arrays...;
    init...,
) where {F, O} = mapreduce(f, op, map(unwrap, arrays)...; init...)

unique_shmem_thread_on_device(f::F, ::AbstractCPUDevice) where {F} = f()
