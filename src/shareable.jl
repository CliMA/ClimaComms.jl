inferred_axes(itr::StaticArrays.SOneTo) = (itr,)
inferred_axes(itr::StaticArrays.StaticArray) = axes(itr)
inferred_axes(itr::Tuple) = (StaticArrays.SOneTo(length(itr)),)
inferred_axes(itr) = nothing

inferred_broadcast_axes(::Nothing, ::Nothing) = nothing
inferred_broadcast_axes(axes1, ::Nothing) = axes1
inferred_broadcast_axes(::Nothing, axes2) = axes2
inferred_broadcast_axes(axes1, axes2) = Broadcast.broadcast_shape(axes1, axes2)

combine_inferred_axes(a) = inferred_axes(a)
combine_inferred_axes(a, b) =
    inferred_broadcast_axes(inferred_axes(a), inferred_axes(b))

inferred_device(::T) where {T} = inferred_device(T)
inferred_device(::Type) = CPUSingleThreaded()

inferred_broadcast_device(::CPUSingleThreaded, ::CPUSingleThreaded) =
    CPUSingleThreaded()
inferred_broadcast_device(device1, ::CPUSingleThreaded) = device1
inferred_broadcast_device(::CPUSingleThreaded, device2) = device2
inferred_broadcast_device(device1, device2) =
    device1 == device2 ? device1 :
    error("Inferred device $device1 cannot be combined with $device2")

combine_inferred_devices(a, b) =
    inferred_broadcast_device(inferred_device(a), inferred_device(b))

inferred_metadata(::T) where {T} = inferred_metadata(T)
inferred_metadata(::Type) = nothing

inferred_broadcast_metadata(::Nothing, ::Nothing) = nothing
inferred_broadcast_metadata(metadata1, ::Nothing) = metadata1
inferred_broadcast_metadata(::Nothing, metadata2) = metadata2
inferred_broadcast_metadata(metadata1, metadata2) =
    metadata1 == metadata2 ? metadata1 :
    error("Inferred metadata $metadata1 cannot be combined with $metadata2")

combine_inferred_metadata(a, b) =
    inferred_broadcast_metadata(inferred_metadata(a), inferred_metadata(b))

"""
    shareable(itr, device)

Wrapper for a [`threadable`](@ref) iterator whose data can be shared across
multiple threads on GPU devices. Most methods for iterators and `AbstractArray`s
are forwarded to the underlying `itr`, but loops and reductions are modified so
that each thread only interacts with a subset of the data in `itr`.

Specifically, when `eachindex` is called on a `shareable` iterator, the threads
in each GPU block evenly divide the available indices among themselves. The
corresponding methods for `map`, `iterate` (which is used by for-loops), and
`copyto!` (which is used by broadcast expressions) all rely on `eachindex` to
specify which items to loop over. Analogues of `reduce!` and `mapreduce!`,
called [`shmem_reduce!`](@ref) and [`shmem_mapreduce!`](@ref), can also be used
to evaluate parallel reductions of `shareable` iterators on GPUs, with runtime
that scales in proportion to the base-2 logarithm of iterator length.

Any broadcast expression over `shareable` iterators can have its output
container automatically allocated in the form of a [`shmem_array`](@ref).
Non-`shareable` iterators can also be used in broadcast expressions with
`shareable` ones, but the resulting output containers may not be inferred
correctly if the non-`shareable` iterators have different dimensions than the
`shareable` ones.

Every `shareable` iterator has an internal `auto_sync` flag that determines
whether [`auto_sync!`](@ref) should be called after every loop or reduction. By
default, this flag is always set to `true`, but in certain situations it can be
set to `false` using [`disable_auto_sync`](@ref), which may improve performance.

Since global memory is typically only used for storing the final outputs of
computations, rather than caching intermediate data like shared memory,
`auto_sync!` only synchronizes threads that modify shared memory arrays. If
global memory arrays are used to cache intermediate data instead of shared
memory arrays, loops and reductions may need to be explicitly synchronized using
[`sync_shmem_threads!`](@ref).
"""
shareable(itr, device, metadata = nothing, auto_sync = true) =
    ShareableIterator(itr, device, metadata, auto_sync)

struct ShareableIterator{I, D, M, A}
    itr::I
end
ShareableIterator(itr::I, device, metadata, auto_sync) where {I} =
    ShareableIterator{I, device, metadata, auto_sync}(itr)

unwrap_type(::Type{<:ShareableIterator{I}}) where {I} = I

unwrap((; itr)::ShareableIterator) = itr
unwrap(itr) = itr

inferred_axes(itr::ShareableIterator) = inferred_axes(unwrap(itr))

inferred_device(::Type{<:ShareableIterator{I, D}}) where {I, D} = D
inferred_metadata(::Type{<:ShareableIterator{I, D, M}}) where {I, D, M} = M

auto_sync_enabled(::ShareableIterator{I, D, M, A}) where {I, D, M, A} = A

auto_unroll_enabled(itr) =
    !isnothing(inferred_axes(itr)) && !(
        needs_metadata_to_unroll_shmem_loops(inferred_device(itr)) &&
        isnothing(inferred_metadata(itr))
    )

unroll_vals(itr) =
    needs_metadata_to_unroll_shmem_loops(inferred_device(itr)) ?
    (Val(length(itr)), Val(inferred_metadata(itr))) : (Val(length(itr)),)

similar_shareable_iterator(itr, unwrapped_itr) = shareable(
    unwrapped_itr,
    inferred_device(itr),
    inferred_metadata(itr),
    auto_sync_enabled(itr),
)

"""
    set_metadata(itr, metadata)

Adds metadata to a [`shareable`](@ref) iterator that can be inferred during
compilation, allowing loops and reductions over the iterator to be unrolled.
"""
set_metadata(itr, metadata) = shareable(
    unwrap(itr),
    inferred_device(itr),
    metadata,
    auto_sync_enabled(itr),
)

"""
    disable_auto_sync(itr)

Sets the `auto_sync` flag of a [`shareable`](@ref) iterator to `false`, so that
the iterator can be used in a loop or reduction without being followed by a call
to [`auto_sync!`](@ref). This is safe to use in the following scenarios:
 - Looping over a `shareable` iterator without writing new data to shared memory
   or global memory that will need to be read by other threads. For example, a
   loop that just saves the final result of a computation to global memory
   should not be synchronized.
 - Running two consecutive loops that independently modify the same values in
   an array. For example, if a loop modifies the value at each index in an
   `array`, like `array .= sin.(array)`, and then another loop modifies the
   value at each index again without reading data from adjacent indices, like
   `array .*= 2`, the first loop over the `array` should not be synchronized.
 - Applying a reduction operator to an array and then only using the result in
   [`@unique_shmem_thread`](@ref) expressions. Without synchronization, the
   result of any reduction operator is only guaranteed to be correct in threads
   that are part of the first warp in each GPU block. The implementation of
   `@unique_shmem_thread` guarantees that it will always choose one of these
   threads, so the expression it executes will have the correct result
   regardless of whether automatic synchronization is disabled. If the result is
   not needed in other threads, the reduction should not be synchronized.
"""
disable_auto_sync(itr) =
    shareable(unwrap(itr), inferred_device(itr), inferred_metadata(itr), false)

"""
    auto_sync!(itr)

Synchronizes GPU threads with access to the same [`shareable`](@ref) iterators,
if the `auto_sync` flag of the given `shareable` iterator is set to `true`. On
CPU devices, this function does nothing. Calls to `auto_sync!` are automatically
inserted into most loops and reductions over `shareable` iterators, but custom
loop functions may require explicit calls to `auto_sync!`.
"""
auto_sync!(itr) =
    auto_sync_enabled(itr) && sync_shmem_threads!(inferred_device(itr))

"""
    sync_shmem_threads!(device)

Internal function called by [`auto_sync!`](@ref) for [`shareable`](@ref)
iterators whose `auto_sync` flags are set to `true`. Non-standard operations
with loops or reductions may require explicit calls to `sync_shmem_threads!`.
"""
function sync_shmem_threads! end

Base.IteratorEltype(::Type{T}) where {T <: ShareableIterator} =
    Base.IteratorEltype(unwrap_type(T))
Base.eltype(::Type{T}) where {T <: ShareableIterator} = eltype(unwrap_type(T))

Base.IteratorSize(::Type{T}) where {T <: ShareableIterator} =
    Base.IteratorSize(unwrap_type(T))
Base.ndims(::Type{T}) where {T <: ShareableIterator} = ndims(unwrap_type(T))
Base.axes(itr::ShareableIterator) = axes(unwrap(itr))
Base.length(itr::ShareableIterator) = length(unwrap(itr))
Base.isempty(itr::ShareableIterator) = isempty(unwrap(itr))
Base.size(itr::ShareableIterator, dim...) = size(unwrap(itr), dim...)

Base.IndexStyle(::Type{T}) where {T <: ShareableIterator} =
    IndexStyle(unwrap_type(T))
Base.firstindex(itr::ShareableIterator) = firstindex(unwrap(itr))
Base.lastindex(itr::ShareableIterator) = lastindex(unwrap(itr))
Base.@propagate_inbounds Base.getindex(itr::ShareableIterator, indices...) =
    getindex(unwrap(itr), indices...)
Base.@propagate_inbounds Base.setindex!(itr::ShareableIterator, x, indices...) =
    setindex!(unwrap(itr), x, indices...)
Base.@propagate_inbounds Base.view(itr::ShareableIterator, indices...) =
    similar_shareable_iterator(itr, view(unwrap(itr), indices...))
Base.similar(itr::ShareableIterator, ::Type{T}, dims...) where {T} =
    similar_shareable_iterator(itr, similar(unwrap(itr), T, dims...))

Base.eachindex(style, itr::ShareableIterator) =
    auto_unroll_enabled(itr) ?
    unrolled_shmem_thread_indices(inferred_device(itr), unroll_vals(itr)...) :
    shmem_thread_indices(inferred_device(itr), length(itr))

function Base.iterate(itr::ShareableIterator, state...)
    index_and_next_state = iterate(eachindex(itr), state...)
    if isnothing(index_and_next_state)
        auto_sync!(itr)
        return nothing
    end
    return @inbounds (itr[index_and_next_state[1]], index_and_next_state[2])
end

Base.map(f::F, args::ShareableIterator...) where {F} = f.(args...)
Base.map!(f::F, dest::ShareableIterator, args...) where {F} =
    dest .= f.(args...)
Base.map!(f::F, dest, args::ShareableIterator...) where {F} =
    dest .= f.(args...)
Base.map!(f::F, dest::ShareableIterator, args::ShareableIterator...) where {F} =
    dest .= f.(args...)

struct ShareableStyle{N, D, M} <: Broadcast.BroadcastStyle end
ShareableStyle{N}(device, metadata) where {N} =
    ShareableStyle{N, device, metadata}()

inferred_device(::Type{<:ShareableStyle{N, D}}) where {N, D} = D
inferred_metadata(::Type{<:ShareableStyle{N, D, M}}) where {N, D, M} = M

Broadcast.broadcastable(itr::ShareableIterator) = itr

Broadcast.BroadcastStyle(::Type{S}) where {S <: ShareableIterator} =
    ShareableStyle{ndims(S)}(inferred_device(S), inferred_metadata(S))

Broadcast.BroadcastStyle(s::ShareableStyle, ::Broadcast.Style{Tuple}) = s

Broadcast.BroadcastStyle(
    s::ShareableStyle{N1},
    ::Broadcast.DefaultArrayStyle{N2},
) where {N1, N2} =
    ShareableStyle{max(N1, N2)}(inferred_device(s), inferred_metadata(s))

Broadcast.BroadcastStyle(
    s1::ShareableStyle{N1},
    s2::ShareableStyle{N2},
) where {N1, N2} = ShareableStyle{max(N1, N2)}(
    combine_inferred_devices(s1, s2),
    combine_inferred_metadata(s1, s2),
)

inferred_axes(bc::Broadcast.Broadcasted) = combine_inferred_axes(bc.args...)
axes_were_inferred(bc) = bc.axes isa Tuple{Vararg{StaticArrays.SOneTo}}

function Broadcast.instantiate(
    bc::Broadcast.Broadcasted{<:ShareableStyle{N}},
) where {N}
    # TODO: Can this check be done efficiently on GPUs?
    # instantiated_axes = if isnothing(bc.axes)
    #     inferred_axes(bc)
    # elseif axes_were_inferred(bc) || length(bc.axes) >= N
    #     Broadcast.check_broadcast_axes(bc.axes, bc.args...)
    #     bc.axes
    # else
    #     StaticArrays.static_check_broadcast_shape(bc.axes, inferred_axes(bc))
    # end
    instantiated_axes =
        isnothing(bc.axes) || length(bc.axes) <= N ? inferred_axes(bc) : bc.axes
    return Broadcast.Broadcasted(bc.style, bc.f, bc.args, instantiated_axes)
end

function Base.similar(
    bc::Broadcast.Broadcasted{<:ShareableStyle},
    ::Type{T},
) where {T}
    axes_were_inferred(bc) ||
        error("Broadcast axes were not inferred during compilation: $(bc.axes)")
    return shmem_array(
        inferred_device(bc.style),
        T,
        map(length, bc.axes),
        inferred_metadata(bc.style),
    )
end

unrolled_foreach(f::F, indices) where {F} =
    unrolled_foreach(f, indices, Val(length(indices)))
@generated unrolled_foreach(f, indices, ::Val{n_indices}) where {n_indices} =
    :(@inline Base.Cartesian.@nexprs $n_indices i -> f(indices[i]))

function shmem_copyto!(dest, src)
    # TODO: Can this check be done efficiently on GPUs?
    # @boundscheck axes(dest) == axes(src) ||
    #              Broadcast.throwdm(axes(dest), axes(src))
    preprocessed_src = src # Broadcast.preprocess(dest, src)
    indices = eachindex(preprocessed_src)
    if auto_unroll_enabled(indices)
        unrolled_foreach(indices) do index
            @inbounds dest[index] = preprocessed_src[index]
        end
    else
        @inbounds @simd for index in indices
            dest[index] = preprocessed_src[index]
        end
    end
    auto_sync!(dest)
    return dest
end

Base.copyto!(dest::ShareableIterator, src) = shmem_copyto!(dest, src)
Base.copyto!(dest, src::ShareableIterator) = shmem_copyto!(dest, src)
Base.copyto!(dest::ShareableIterator, src::ShareableIterator) =
    shmem_copyto!(dest, src)
Base.copyto!(dest, src::Broadcast.Broadcasted{<:ShareableStyle}) =
    shmem_copyto!(dest, src)
Base.copyto!(
    dest::ShareableIterator,
    src::Broadcast.Broadcasted{<:ShareableStyle},
) = shmem_copyto!(dest, src)

"""
    shmem_array(itr, [T], [dims])
    shmem_array(device, T, dims)

Device-flexible array whose element type `T` and dimensions `dims` are known
during compilation, which corresponds to a static shared memory array on a GPU.
Shared memory can be jointly modified by the threads in each GPU block, allowing
kernels to have interdependencies between threads. On CPU devices, which do not
provide access to high-performance shared memory, a `shmem_array` instead
corresponds to a statically-sized `MArray` stored in register memory. Shared
memory on a GPU and register memory on a CPU are both much faster to access and
modify than the global memory used by `Array`s and `CuArray`s.

The recommended method for this function uses a [`shareable`](@ref) iterator to
determine the device, a default element type, and a default array size, and an
alternative method that directly accepts those parameters is also available.
However, the alternative method does not propagate additional metadata from the
`shareable` iterator that is required for unrolling of loops and reductions, so
using it disables automatic unrolling of operations on the resulting array.
If the default element type or dimensions of a `shareable` iterator cannot be
inferred during compilation, inferrable values of `T` or `dims` need to be
passed to the `shmem_array` function to override the default values.
"""
shmem_array(itr) = shmem_array(itr, eltype(itr), size(itr))
shmem_array(itr, dims) = shmem_array(itr, eltype(itr), dims)
shmem_array(itr, ::Type{T}) where {T} = shmem_array(itr, T, size(itr))
shmem_array(itr, ::Type{T}, dims) where {T} =
    shmem_array(inferred_device(itr), T, dims, inferred_metadata(itr))
shmem_array(device::AbstractDevice, ::Type{T}, dims, metadata...) where {T} =
    shareable(unwrapped_shmem_array(device, T, dims), device, metadata...)

"""
    @unique_shmem_thread itr ...

Isolates an expression that should only be evaluated on a single thread in a
[`@threaded`](@ref) loop with an [`@shmem_threaded`](@ref) iterator. For
example, the result of a reduction over shared memory arrays can only be written
to some specific address in global memory within a `@unique_shmem_thread`
expression, since multiple threads simultaneously writing to the same index can
lead to race conditions.

Any call to [`auto_sync!`](@ref) in a `@unique_shmem_thread` expression can lead
to deadlock, as the thread that issues a request for synchronization will be
stuck waiting for other threads that never issue one. Since loops and reductions
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
unique_shmem_thread(f::F, itr) where {F} =
    unique_shmem_thread(f, inferred_device(itr))

"""
    shmem_reduce!(op, [dest], itr; [init])

Device-flexible version of `reduce(op, itr; [init])`, where `itr` is a
[`shareable`](@ref) iterator that supports `setindex!`. When `dest` is also
specified, the reduced value is written to it; otherwise, the value is returned
at the end of the function.

On GPU devices, `shmem_reduce!` is parallelized across the threads in each
block, so that its runtime that is roughly proportional to `log(length(itr))`,
as long as the block size is no smaller than `length(itr)`. To enable this
parallelization, the `itr` is used as a cache for intermediate data, so the
values in it should not be used after this function is called. On CPU devices,
`shmem_reduce!` is equivalent to the standard `reduce` function, and the `itr`
is not overwritten. The order in which `shmem_reduce!` evaluates `op` is
device-dependent, so reductions of floating-point values are not always
guaranteed to be bitwise reproducible.
"""
shmem_reduce!(op::O, itr; init...) where {O} =
    reduce_on_device!(op, inferred_device(itr), itr; init...)
shmem_reduce!(op::O, dest, itr; init...) where {O} =
    reduce_on_device!(op, inferred_device(itr), dest, itr; init...)

function reduce_in_place_on_device!(op::O, device, itr) where {O}
    isempty(itr) && return
    auto_unroll_enabled(itr) ?
    unrolled_reduce_in_place!(op, device, unwrap(itr), unroll_vals(itr)...) :
    reduce_in_place!(op, device, unwrap(itr))
end

function in_place_reduced_value(itr; init...)
    keys(init) in ((), (:init,)) ||
        error("The only keyword argument supported by shmem_reduce! is `init`")
    (isempty(itr) && isempty(init)) &&
        error("Reduction of an empty collection requires an `init` value")
    return @inbounds isempty(itr) ? values(init)[1] :
                     (isempty(init) ? itr[1] : op(values(init)[1], itr[1]))
end

function reduce_on_device!(op::O, device, itr; init...) where {O}
    reduce_in_place_on_device!(op, device, itr)
    !isempty(itr) && auto_sync!(itr)
    return in_place_reduced_value(itr; init...)
end
function reduce_on_device!(op::O, device, dest, itr; init...) where {O}
    reduce_in_place_on_device!(op, device, itr)
    @unique_shmem_thread itr dest[] = in_place_reduced_value(itr; init...)
end

"""
    shmem_mapreduce!(f, op, [dest], itr; [init])

Device-flexible version of `mapreduce(f, op, itr; [init])`, where `itr` is a
[`shareable`](@ref) iterator that supports `setindex!`. When `dest` is also
specified, the reduced value is written to it; otherwise, the value is returned
at the end of the function.

On GPU devices, `shmem_mapreduce!` is a combination of `map!(f, itr, itr)` and
`shmem_reduce!(op, itr; init)`. On CPU devices, `shmem_mapreduce!` is equivalent
to the standard `mapreduce` function, and the `itr` is not overwritten.
"""
shmem_mapreduce!(f::F, op::O, itr; init...) where {F, O} =
    mapreduce_on_device!(f, op, inferred_device(itr), itr; init...)
shmem_mapreduce!(f::F, op::O, dest, itr; init...) where {F, O} =
    mapreduce_on_device!(f, op, inferred_device(itr), dest, itr; init...)

mapreduce_on_device!(f::F, op::O, device, itr; init...) where {F, O} =
    reduce_on_device!(op, device, map!(f, itr, itr); init...)
mapreduce_on_device!(f::F, op::O, device, dest, itr; init...) where {F, O} =
    reduce_on_device!(op, device, dest, map!(f, itr, itr); init...)

for (func, op, init) in ((:any, :|, :false), (:all, :&, :true))
    shmem_func! = Symbol(:shmem_, func, :!)
    @eval begin
        """
            $($(string(shmem_func!)))([f], [dest], itr)

        Device-flexible version of `$($(string(func)))([f], itr)`, equivalent
        to calling [`shmem_reduce!`](@ref) or [`shmem_mapreduce!`](@ref) with
        the `$($(string(op)))` operator.
        """
        @inline $shmem_func!(arrays...) =
            $shmem_reduce!($op, arrays...; init = $init)
        @inline $shmem_func!(f::F, arrays...) where {F <: Function} =
            $shmem_mapreduce!(f, $op, arrays...; init = $init)
    end
end

for (func, op) in ((:sum, :+), (:prod, :*), (:maximum, :max), (:minimum, :min))
    shmem_func! = Symbol(:shmem_, func, :!)
    @eval begin
        """
            $($(string(shmem_func!)))([f], [dest], itr; [init])

        Device-flexible version of `$($(string(func)))([f], itr; [init])`,
        equivalent to calling [`shmem_reduce!`](@ref) or
        [`shmem_mapreduce!`](@ref) with the `$($(string(op)))` operator.
        """
        @inline $shmem_func!(arrays...; init...) =
            $shmem_reduce!($op, arrays...; init...)
        @inline $shmem_func!(f::F, arrays...; init...) where {F <: Function} =
            $shmem_mapreduce!(f, $op, arrays...; init...)
    end
end
