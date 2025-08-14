"""
    synchronize_shmem!(device)

Synchronizes GPU threads with access to the same shared memory arrays. On CPU
devices, this function does nothing. Calls to `synchronize_shmem!` are
automatically inserted into most loops and reductions over shared memory arrays.
"""
function synchronize_shmem! end

abstract type WrappedIterator{I} end

unwrap((; itr)::WrappedIterator) = itr

Base.IteratorEltype(::Type{<:WrappedIterator{I}}) where {I} =
    Base.IteratorEltype(I)
Base.IteratorSize(::Type{<:WrappedIterator{I}}) where {I} = Base.IteratorSize(I)
Base.eltype(::Type{<:WrappedIterator{I}}) where {I} = eltype(I)
Base.length(itr::WrappedIterator) = length(unwrap(itr))
Base.size(itr::WrappedIterator, dims...) = size(unwrap(itr), dims...)
Base.isempty(itr::WrappedIterator) = isempty(unwrap(itr))
Base.iterate(itr::WrappedIterator, state...) = iterate(unwrap(itr), state...)

struct SyncableIterator{I, D, A} <: WrappedIterator{I}
    itr::I
    device::D
end
SyncableIterator(itr::I, device::D, A) where {I, D} =
    SyncableIterator{I, D, A}(itr, device)

auto_sync_flag(::SyncableIterator{I, D, A}) where {I, D, A} = A
device_to_sync((; device)::SyncableIterator) = device

"""
    syncable(itr, device, [auto_sync])

Modifies an iterator so that every loop and reduction over it is followed by a
call to [`synchronize_shmem!`](@ref) with the specified `device`, unless the
`auto_sync` flag is changed to `false`.
"""
syncable(itr, device, auto_sync = true) =
    SyncableIterator(itr, device, auto_sync)

"""
    unsync(itr)

Sets the `auto_sync` flag of a [`syncable`](@ref) iterator to `false`, so that
the iterator can be used in a loop or reduction without being followed by a call
to [`synchronize_shmem!`](@ref). This is safe to use in the following scenarios:
 - Looping over a [`@shmem_threaded`](@ref) iterator without writing new data to
   shared memory or global memory that will need to be read by other threads.
   For example, any loop that saves the final result of a computation to global
   memory does not need to be synchronized. It is also safe to avoid automatic
   synchronization when two consecutive loops independently modify the same
   values in shared memory or global memory, e.g., when a loop modifies
   `array[i]` for each index `i` in a `@shmem_threaded` iterator, followed by
   another loop that also only reads and modifies `array[i]` for each index `i`.
 - Applying a reduction operator to a [`shmem_array`] and then only using the
   result inside [`@unique_shmem_thread`](@ref) expressions. Without automatic
   synchronization, the result of any reduction operator is only guaranteed to
   be correct in threads that are part of the first warp in each GPU block. The
   implementation of `@unique_shmem_thread` guarantees that it will always
   choose one of these threads, so the expression it executes will have the
   correct result regardless of whether automatic synchronization is disabled.
"""
unsync(itr::SyncableIterator) =
    syncable(unwrap(itr), device_to_sync(itr), false)
unsync(itr) = itr

"""
    auto_synchronize!(itr)

Calls [`synchronize_shmem!`](@ref) on the `device` of a [`syncable`](@ref)
iterator if its `auto_sync` flag is set to `true`.
"""
auto_synchronize!(itr) =
    auto_sync_flag(itr) && synchronize_shmem!(device_to_sync(itr))

function Base.iterate(itr::SyncableIterator, state...)
    result = iterate(unwrap(itr), state...)
    isnothing(result) && auto_synchronize!(itr)
    return result
end

Base.IndexStyle(::Type{<:SyncableIterator{I}}) where {I} = IndexStyle(I)
Base.firstindex(itr::SyncableIterator) = firstindex(unwrap(itr))
Base.lastindex(itr::SyncableIterator) = lastindex(unwrap(itr))
Base.axes(itr::SyncableIterator, dim...) = axes(unwrap(itr), dim...)
Base.@propagate_inbounds Base.getindex(itr::SyncableIterator, is...) =
    getindex(unwrap(itr), is...)
Base.@propagate_inbounds Base.setindex!(itr::SyncableIterator, v, is...) =
    setindex!(unwrap(itr), v, is...)
Base.@propagate_inbounds Base.view(itr::SyncableIterator, is...) =
    syncable(view(unwrap(itr), is...), device_to_sync(itr), auto_sync_flag(itr))
Base.similar(itr::SyncableIterator, ::Type{T}, dims...) where {T} = syncable(
    similar(unwrap(itr), T, dims...),
    device_to_sync(itr),
    auto_sync_flag(itr),
)

struct StaticSizeIterator{I, S} <: WrappedIterator{I}
    itr::I
end
StaticSizeIterator(itr::I, S) where {I} = StaticSizeIterator{I, S}(itr)

"""
    static_resize(itr, size_val)

Overrides the `size` of an iterator with the specified value, which is wrapped
in a `Val` so that it can be inferred during compilation. The `length` of the
iterator is also modified accordingly. None of the iterator's underlying data is
modified, though, so the new iterator should not be passed to any functions that
expect consistency between the iterator's size and data, like `axes` and `view`.
"""
static_resize(itr, ::Val{S}) where {S} = StaticSizeIterator(itr, S)

Base.size(::StaticSizeIterator{I, S}) where {I, S} = S

Base.length(itr::StaticSizeIterator) = prod(size(itr))
Base.size(itr::StaticSizeIterator, dim) = size(itr)[dim]
Base.isempty(itr::StaticSizeIterator) = length(itr) == 0

"""
    shmem_array(itr, T, [dims...])
    shmem_array(device, T, dims...)

Device-flexible array whose element type `T` and dimensions `dims` are known
during compilation, which corresponds to a static shared memory array on a GPU.
Shared memory can be jointly modified by the threads in each GPU block, allowing
kernels to have interdependencies between threads. On CPU devices, which do not
provide access to high-performance shared memory, a `shmem_array` instead
corresponds to a statically-sized `MArray` stored in register memory. Shared
memory on a GPU and register memory on a CPU are both much faster to access and
modify than the global memory used for `Array`s and `CuArray`s.

The recommended method for this function uses a [`syncable`](@ref) iterator to
determine the device and a default array size, but an alternative method that
directly takes the device as an argument is also available. If a `syncable`
iterator is modified with [`unsync`](@ref), any `shmem_array` constructed from
it will have automatic thread synchronization at the end of reductions disabled.
"""
shmem_array(itr::SyncableIterator, ::Type{T}, dims...) where {T} = syncable(
    shmem_array_on_device(device_to_sync(itr), T, dims...),
    device_to_sync(itr),
    auto_sync_flag(itr),
)
shmem_array(itr::SyncableIterator, ::Type{T}) where {T} =
    shmem_array(itr, T, size(itr)...)

shmem_array(device::AbstractDevice, ::Type{T}, dims...) where {T} =
    syncable(shmem_array_on_device(device, T, dims...), device)
shmem_array(_, ::Type{T}, dims...) where {T} =
    shmem_array(CPUSingleThreaded(), T, dims...)

"""
    shmem_map!(f, dest_array, src_arrays...)

Device-flexible version of `map!(f, dest_array, src_arrays...)`, where
`dest_array` and the `src_arrays` are results of [`shmem_array`](@ref). On GPU
devices, this is parallelized across the threads in each block, so that its
runtime is approximately constant with respect to `length(dest_array)`.
"""
shmem_map!(f::F, dest_array, src_arrays...) where {F} = shmem_map_on_device!(
    f,
    device_to_sync(dest_array),
    dest_array,
    src_arrays...,
)

shmem_map_on_device!(f::F, device, dest_array, src_arrays...) where {F} =
    shmem_map_on_device!(splat(f), device, dest_array, zip(src_arrays...))

"""
    shmem_reduce!(op, array; [init])

Device-flexible version of `reduce(op, array; [init])`, where the `array` is a
result of [`shmem_array`](@ref). On GPU devices, this is parallelized across the
threads in each block, so that its runtime is approximately proportional to
`log(length(array))` as long as `length(array)` is less than the number of
threads per block. To enable this parallelization, the `array` also acts as a
cache for intermediate data. On CPU devices, `shmem_reduce!` is equivalent to
the standard `reduce` function, and the `array` is not overwritten. The order in
which `shmem_reduce!` evaluates `op` is device-dependent, so reductions of
floating-point values are not always guaranteed to be bitwise reproducible.
"""
shmem_reduce!(op::O, array; init...) where {O} =
    shmem_reduce_on_device!(op, device_to_sync(array), array; init...)

"""
    shmem_mapreduce!(f, op, arrays...; [cache], [init])

Device-flexible version of `mapreduce(f, op, arrays...; [init])`, where the
`arrays` are results of [`shmem_array`](@ref). On GPU devices, this is a
combination of `shmem_map!(f, cache, arrays...)` and
`shmem_reduce(op, cache; init)`, where the first of the `arrays` is used as the
`cache` by default. On CPU devices, this is instead just a call to `mapreduce`,
and the `cache` is not used. The order in which `shmem_mapreduce!` evaluates
`op` is device-dependent, so reductions of floating-point values are not always
guaranteed to be bitwise reproducible.
"""
shmem_mapreduce!(
    f::F,
    op::O,
    array,
    arrays...;
    cache = array,
    init...,
) where {F, O} = shmem_mapreduce_on_device!(
    f,
    op,
    device_to_sync(cache),
    cache,
    array,
    arrays...;
    init...,
)

function shmem_mapreduce_on_device!(
    f::F,
    op::O,
    device,
    cache,
    arrays...;
    init...,
) where {F, O}
    shmem_map_on_device!(f, device, cache, arrays...)
    return shmem_reduce_on_device!(op, device, cache; init...)
end

for (func, op, init) in ((:any, :|, :false), (:all, :&, :true))
    shmem_func! = Symbol(:shmem_, func, :!)
    @eval begin
        """
            $($(string(shmem_func!)))([f], array)

        Device-flexible version of `$($(string(func)))([f], array)`,
        equivalent to [`shmem_reduce!`](@ref) or [`shmem_mapreduce!`](@ref) with
        the reduction operator `$($(string(op)))`.
        """
        $shmem_func!(f::F, array) where {F} =
            $shmem_mapreduce!(f, $op, array; init = $init)
        $shmem_func!(array) = $shmem_reduce!($op, array; init = $init)
    end
end

for (func, op) in ((:sum, :+), (:prod, :*), (:maximum, :max), (:minimum, :min))
    shmem_func! = Symbol(:shmem_, func, :!)
    @eval begin
        """
            $($(string(shmem_func!)))([f], array; [init])

        Device-flexible version of `$($(string(func)))([f], array; [init])`,
        equivalent to [`shmem_reduce!`](@ref) or [`shmem_mapreduce!`](@ref) with
        the reduction operator `$($(string(op)))`.
        """
        $shmem_func!(f::F, array; init...) where {F} =
            $shmem_mapreduce!(f, $op, array; init...)
        $shmem_func!(array; init...) = $shmem_reduce!($op, array; init...)
    end
end

"""
    @unique_shmem_thread itr ...

Isolates an expression that should only be evaluated on a single thread in a
[`@threaded`](@ref) loop with an [`@shmem_threaded`](@ref) iterator. For
example, the result of a reduction over shared memory arrays can only be written
to some specific address in global memory within a `@unique_shmem_thread`
expression, since multiple threads simultaneously writing to the same index can
lead to race conditions.

Any call to [`synchronize_shmem!`](@ref) in a `@unique_shmem_thread` expression
will lead to deadlock, as the thread that calls `synchronize_shmem!` will be
stuck waiting for other threads that never call it. Since loops and reductions
over shared memory generally require thread synchronization, they should not be
used in `@unique_shmem_thread` expressions.
"""
macro unique_shmem_thread(itr, expr)
    return :(unique_shmem_thread(() -> $(esc(expr)), $(esc(itr))))
end

"""
    unique_shmem_thread(f, itr)

Functional form of [`@unique_shmem_thread`](@ref), equivalent to the expression
`@unique_shmem_thread itr f()`.
"""
unique_shmem_thread(f::F, itr::SyncableIterator) where {F} =
    unique_shmem_thread_on_device(f, device_to_sync(itr))
unique_shmem_thread(f::F, _) where {F} =
    unique_shmem_thread_on_device(f, CPUSingleThreaded())
