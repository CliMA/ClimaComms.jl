"""
    threadable(device, itrs...)

Modifies one or more iterators so that they each have the following properties:
 - a concrete element type
 - a well-defined `length`, and a corresponding method for `isempty`
 - support for linear indexing through `getindex`, with `firstindex` set to 1
When no specialized method of `threadable` is defined for some iterator type, an
error will be thrown if the iterator does not already have these properties.

The `threadable` version of `Iterators.product` internally converts linear
indices to Cartesian indices with `Base.multiplicativeinverse` on GPU devices,
avoiding inefficient division operations. On CPU devices, this conversion is
done using regular integer division.
"""
function threadable(_, itr)
    throw_threadable_error(err) =
        throw(ArgumentError("$(typeof(itr)) is not threadable because it $err"))
    Base.IteratorEltype(typeof(itr)) isa Base.HasEltype &&
    isconcretetype(eltype(itr)) ||
        throw_threadable_error("does not have a concrete element type")
    Base.IteratorSize(typeof(itr)) isa Union{Base.HasLength, Base.HasShape} ||
        throw_threadable_error("does not have a well-defined length")
    firstindex(itr) == 1 ||
        throw_threadable_error("does not support one-based linear indexing")
    return itr
end

threadable(device, itrs...) = map(Base.Fix1(threadable, device), itrs)

abstract type ThreadableIterator end
Base.firstindex(::ThreadableIterator) = 1
Base.lastindex(itr::ThreadableIterator) = length(itr)
Base.isempty(itr::ThreadableIterator) = length(itr) == 0

struct ThreadableGenerator{F, I} <: ThreadableIterator
    f::F
    itr::I
end
Adapt.@adapt_structure ThreadableGenerator
threadable(device, (; f, iter)::Base.Generator) =
    ThreadableGenerator(f, threadable(device, iter))
Base.length((; itr)::ThreadableGenerator) = length(itr)
Base.@propagate_inbounds Base.getindex(
    (; f, itr)::ThreadableGenerator,
    i::Integer,
) = f(itr[i])

struct ThreadableReverse{I} <: ThreadableIterator
    itr::I
end
Adapt.@adapt_structure ThreadableReverse
threadable(device, (; itr)::Iterators.Reverse) =
    ThreadableReverse(threadable(device, itr))
Base.length((; itr)::ThreadableReverse) = length(itr)
Base.@propagate_inbounds Base.getindex((; itr)::ThreadableReverse, i::Integer) =
    itr[length(itr) - i + 1]

struct ThreadableEnumerate{I} <: ThreadableIterator
    itr::I
end
Adapt.@adapt_structure ThreadableEnumerate
threadable(device, (; itr)::Iterators.Enumerate) =
    ThreadableEnumerate(threadable(device, itr))
Base.length((; itr)::ThreadableEnumerate) = length(itr)
Base.@propagate_inbounds Base.getindex(
    (; itr)::ThreadableEnumerate,
    i::Integer,
) = (i, itr[i])

struct ThreadableZip{I} <: ThreadableIterator
    itrs::I
end
Adapt.@adapt_structure ThreadableZip
threadable(device, (; is)::Iterators.Zip) =
    ThreadableZip(threadable(device, is...))
Base.length((; itrs)::ThreadableZip) = minimum(length, itrs)
Base.@propagate_inbounds Base.getindex((; itrs)::ThreadableZip, i::Integer) =
    map(Base.Fix2(getindex, i), itrs)

struct ThreadableFlatten{I, O} <: ThreadableIterator
    itrs::I
    offsets::O
end
Adapt.@adapt_structure ThreadableFlatten
function threadable(device, (; it)::Iterators.Flatten)
    itrs = threadable(device, threadable(device, it)...)
    offsets = cumsum((0, map(length, itrs[1:(end - 1)])...))
    return ThreadableFlatten(itrs, offsets)
end
Base.length((; itrs)::ThreadableFlatten) = sum(length, itrs)
Base.@propagate_inbounds function Base.getindex(
    (; itrs, offsets)::ThreadableFlatten,
    i::Integer,
)
    itr_index = findlast(<(i), offsets)::Integer
    return itrs[itr_index][i - offsets[itr_index]]
end

struct ThreadablePartition{I} <: ThreadableIterator
    itr::I
    partition_size::Int
end
Adapt.@adapt_structure ThreadablePartition
threadable(device, (; c, n)::Iterators.PartitionIterator) =
    ThreadablePartition(threadable(device, c), n)
Base.length((; itr, partition_size)::ThreadablePartition) =
    cld(length(itr), partition_size)
Base.@propagate_inbounds function Base.getindex(
    (; itr, partition_size)::ThreadablePartition,
    i::Integer,
)
    first_index_in_partition = (i - 1) * partition_size + 1
    last_index_in_partition = min(i * partition_size, length(itr))
    indices_in_partition = first_index_in_partition:last_index_in_partition
    return Iterators.map(Base.Fix1(getindex, itr), indices_in_partition)
end

struct ThreadableProduct{I, D} <: ThreadableIterator
    itrs::I
    divisors::D
end
Adapt.@adapt_structure ThreadableProduct
function threadable(device, (; iterators)::Iterators.ProductIterator)
    itrs = threadable(device, iterators...)
    divisors = reverse(cumprod((1, map(length, itrs[1:(end - 1)])...)))
    device_optimized_divisors =
        device isa AbstractCPUDevice ? divisors :
        map(Base.multiplicativeinverse, divisors)
    return ThreadableProduct(itrs, device_optimized_divisors)
end
Base.length((; itrs)::ThreadableProduct) = prod(length, itrs)
Base.@propagate_inbounds function Base.getindex(
    (; itrs, divisors)::ThreadableProduct,
    i::Integer,
)
    reversed_offset_remainder_pairs =
        accumulate(divisors; init = (0, i - 1)) do (_, remainder), divisor
            divrem(remainder, divisor)
        end
    offsets = map(first, reverse(reversed_offset_remainder_pairs))
    return map((itr, offset) -> itr[offset + 1], itrs, offsets)
end

# TODO: Check whether converting every Int to an Int32 and unrolling functions
# over tuples improves the performance of getindex on GPU devices.
