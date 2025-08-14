"""
    @threaded [device] [kwargs...] for ... end

Device-flexible generalization of `Threads.@threads`, which distributes the
iterations of a for-loop across multiple threads, with the option to control
thread coarsening and GPU kernel configuration. Coarsening makes each thread
evaluate more than one iteration of the loop, which can improve performance by
reducing the runtime overhead of launching additional threads (though too much
coarsening worsens performance because it reduces parallelization). The `device`
is either inferred by calling `ClimaComms.device()`, or it can be specified
manually, with the following device-dependent behavior:

 - When `device` is a `CPUSingleThreaded()`, the loop is evaluated as-is. This
   avoids the runtime overhead of calling `Threads.@threads` with a single
   thread, and, when the device type is statically inferrable, it also avoids
   compilation overhead.

 - When `device` is a `CPUMultiThreaded()`, the loop is passed to
   `Threads.@threads`. This supports three different kinds of "schedulers" for
   determining how many iterations of the loop to evaluate in each thread:
     1) (default) a "dynamic" scheduler that changes the number of iterations as
        new threads are launched,
     2) a "static" scheduler that evaluates a fixed number of iterations per
        thread, and
     3) a "greedy" scheduler that uses a small number of threads, continuously
        evaluating iterations in each thread until the loop is completed (only
        available as of Julia 1.11).
   Setting `coarsen` to `:dynamic` or `:greedy` launches threads with those
   schedulers. Setting it to `:static` or an integer value launches threads with
   static scheduling (using `:static` is similar to using `1`, but slightly more
   performant). To read more about multi-threading, see the documentation for
   [`Threads.@threads`](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.@threads).

 - When `device` is a `CUDADevice()`, the loop is compiled with `CUDA.@cuda` and
   run with `CUDA.@sync`. Since CUDA launches all threads at the same time, only
   static scheduling can be used. Setting `coarsen` to any symbol causes each
   thread to evaluate a single iteration (default), and setting it to an integer
   value causes each thread to evaluate that number of iterations (the default
   is similar to using `1`, but slightly more performant). If the total number
   of iterations in the loop is extremely large, the specified coarsening may
   require more threads than can be simultaneously launched on the GPU, in which
   case the amount of coarsening is automatically increased.

   The optional argument `block_size` is also available for manually specifying
   the size of each block on a GPU. The default value of `:auto` sets the number
   of threads in each block to the largest possible value that permits a high
   GPU "occupancy" (the number of active thread warps in each multiprocessor
   executing the kernel). An integer can be used instead of `:auto` to override
   this default value. If the specified value exceeds the total number of
   threads, it is automatically decreased to avoid idle threads.

Any iterator with methods for `length` and `getindex` can be used in a
`@threaded` loop, as long as `getindex` supports one-based linear indexing. All
lazy iterators from `Base` and `Base.Iterators`, such as `zip`, `enumerate`,
`Iterators.product`, and generator expressions, are also compatible with
`@threaded`. (Although these iterators do not define methods for `getindex`,
they are automatically modified by [`threadable`](@ref) to support `getindex`.)
Using multiple iterators with `@threaded` is equivalent to looping over a single
`Iterators.product`, with the innermost iterator of the loop appearing first in
the product, and the outermost iterator appearing last.

NOTE: When a value in the body of the loop has a type that cannot be inferred by
the compiler, an `InvalidIRError` will be thrown during compilation for a
`CUDADevice()`. In particular, global variables are not inferrable, so
`@threaded` must be wrapped in a function whenever it is used in the REPL:

```julia-repl
julia> a = CUDA.CuArray{Int}(undef, 100); b = similar(a);

julia> threaded_copyto!(a, b) = ClimaComms.@threaded for i in axes(a, 1)
           a[i] = b[i]
       end
threaded_copyto! (generic function with 1 method)

julia> threaded_copyto!(a, b)

julia> ClimaComms.@threaded for i in axes(a, 1)
           a[i] = b[i]
       end
ERROR: InvalidIRError: ...
```

Moreover, type variables are not inferrable across function boundaries, so types
used in a threaded loop cannot be precomputed before the loop:

```julia-repl
julia> threaded_add_epsilon!(a) = ClimaComms.@threaded for i in axes(a, 1)
           FT = eltype(a)
           a[i] += eps(FT)
       end
threaded_add_epsilon! (generic function with 1 method)

julia> threaded_add_epsilon!(a)

julia> function threaded_add_epsilon!(a)
           FT = eltype(a)
           ClimaComms.@threaded for i in axes(a, 1)
               a[i] += eps(FT)
           end
       end
threaded_add_epsilon! (generic function with 1 method)

julia> threaded_add_epsilon!(a)
ERROR: InvalidIRError: ...
```

To fix other kinds of inference issues on GPU devices, especially ones brought
about by indexing into iterators with nonuniform element types, see
[`UnrolledUtilities.jl`](https://github.com/CliMA/UnrolledUtilities.jl).
"""
macro threaded(args...)
    usage_error_string = """Invalid @threaded syntax; supported usages are
                            - @threaded [device] [kwargs...] for ... end
                            - @threaded [device] [kwargs...] begin
                                  [@shmem_threaded ... = ...]
                                  for ... end
                              end
                            - @threaded [device] [kwargs...] begin
                                  @shmem_threaded ... = ...
                                  begin ... end
                              end"""

    isempty(args) && throw(ArgumentError(usage_error_string))
    (device_and_kwarg_exprs..., loop_or_block_expr) = Any[args...]

    if all(expr -> Meta.isexpr(expr, :(=)), device_and_kwarg_exprs)
        device_expr = :($ClimaComms.device())
        kwarg_exprs = device_and_kwarg_exprs
    elseif all(expr -> Meta.isexpr(expr, :(=)), device_and_kwarg_exprs[2:end])
        (device_expr, kwarg_exprs...) = device_and_kwarg_exprs
    else
        throw(ArgumentError(usage_error_string))
    end
    for kwarg_expr in kwarg_exprs
        kwarg_expr.args[2] isa QuoteNode || continue
        kwarg_expr.args[2] = :(Val($(kwarg_expr.args[2])))
    end

    if Meta.isexpr(loop_or_block_expr, :for)
        macro_expr = nothing
        threaded_expr = loop_or_block_expr
        (loop_assignments_expr, loop_body_expr) = threaded_expr.args
    elseif (
        Meta.isexpr(loop_or_block_expr, :block) &&
        (
            length(loop_or_block_expr.args) == 2 ||
            length(loop_or_block_expr.args) == 4 &&
            Meta.isexpr(loop_or_block_expr.args[2], :macrocall)
        ) &&
        Meta.isexpr(loop_or_block_expr.args[end], :for)
    )
        macro_expr =
            length(loop_or_block_expr.args) == 2 ? nothing :
            loop_or_block_expr.args[2]
        threaded_expr = loop_or_block_expr.args[end]
        (loop_assignments_expr, loop_body_expr) = threaded_expr.args
    elseif (
        Meta.isexpr(loop_or_block_expr, :block) &&
        length(loop_or_block_expr.args) == 4 &&
        Meta.isexpr(loop_or_block_expr.args[2], :macrocall) &&
        Meta.isexpr(loop_or_block_expr.args[4], :block)
    )
        (_, macro_expr, _, threaded_expr) = loop_or_block_expr.args
        loop_assignments_expr = Expr(:block)
        loop_body_expr = threaded_expr
    else
        throw(ArgumentError(usage_error_string))
    end
    assignment_exprs =
        Meta.isexpr(loop_assignments_expr, :block) ?
        loop_assignments_expr.args : Any[loop_assignments_expr]

    # Reverse the order of iterators to match standard for-loop behavior:
    # for-loops go from outermost iterator to innermost iterator, whereas
    # Iterators.product goes from innermost iterator to outermost iterator.
    reversed_assignment_exprs = Iterators.reverse(assignment_exprs)
    var_exprs = Base.mapany(expr -> expr.args[1], reversed_assignment_exprs)
    itr_exprs = Base.mapany(expr -> expr.args[2], reversed_assignment_exprs)

    if isnothing(macro_expr)
        single_threaded_expr = threaded_expr
    elseif (
        (
            macro_expr.args[1] == Symbol("@shmem_threaded") ||
            Meta.isexpr(macro_expr.args[1], :.) &&
            macro_expr.args[1].args[2] == QuoteNode(Symbol("@shmem_threaded"))
        ) &&
        (length(macro_expr.args) == 3 && Meta.isexpr(macro_expr.args[3], :(=)))
    )
        shmem_assignment_expr = macro_expr.args[3]
        (shmem_var_expr, shmem_itr_expr) = shmem_assignment_expr.args
        push!(var_exprs, shmem_var_expr)
        push!(kwarg_exprs, :(shmem_threaded = $shmem_itr_expr))
        single_threaded_expr = :($shmem_assignment_expr; $threaded_expr)
    else
        throw(ArgumentError(usage_error_string))
    end

    # Reduce latency by inlining the expression for single-threaded devices.
    return quote
        device = $(esc(device_expr))
        device isa CPUSingleThreaded ? $(esc(single_threaded_expr)) :
        threaded(
            ($(Base.mapany(esc, var_exprs)...),) -> $(esc(loop_body_expr)),
            device,
            $(Base.mapany(esc, itr_exprs)...);
            $(Base.mapany(esc, kwarg_exprs)...),
        )
    end
end

"""
    @shmem_threaded ... = ...

Extension of [`@threaded`](@ref) for working with shared memory arrays, which
can be used immediately before a `@threaded` loop in the following manner:

```julia
outer_iterator_1 = ...
outer_iterator_2 = ...
...
@threaded device ... begin
    @shmem_threaded inner_iterator = ...
    for outer_item_1 in outer_iterator_1, outer_item_2 in outer_iterator_2, ...
        ...
        array_1 = shmem_array(device, ...)
        array_2 = shmem_array(device, ...)
        ...
        for inner_item in inner_iterator
            <write to arrays based on inner and outer loop items>
        end
        ...
        for inner_item in inner_iterator
            <read from and write to arrays based on inner and outer loop items>
        end
        ...
        for inner_item in inner_iterator
            <read from arrays based on inner and outer loop items>
        end
        ...
    end
end
```

When an iterator variable assignment is annotated with `@shmem_threaded`, the
iterator's elements are distributed across the threads in each block of a GPU,
which have access to the same shared memory arrays. Iterators in the outer
`@threaded` loop are then only distributed across GPU blocks, instead of being
distributed across both blocks and threads within the blocks. On CPU devices,
which do not provide blocks of threads with access to high-performance shared
memory, the `@shmem_threaded` annotation does nothing.

An iterator whose assignment is annotated with `@shmem_threaded` can be used in
loops nested inside the outer `@threaded` loop, with each GPU thread only
processing a subset of the iterator's elements. Coarsening of the inner loops
can be controlled by specifying the `shmem_coarsen` keyword argument for
`@threaded`. Setting `shmem_coarsen` to `:auto` causes each thread to process
one element of the iterator, and setting it to an integer value causes each
thread to process that number of elements.

When the outer loop is not needed, it can be replaced by a `begin ... end`
expression that is only distributed across the threads in a single GPU block:

```julia
@threaded device ... begin
    @shmem_threaded iterator = ...
    begin ... end
end
```

Since an iterator annotated with `@shmem_threaded` can be used to read data from
shared memory arrays, every loop and reduction over the iterator is followed by
a call to [`auto_synchronize!`](@ref) on GPU devices, ensuring that shared
memory reads are consistent across threads.

When synchronization is not needed after a loop over an iterator annotated with
`@shmem_threaded`, it can be disabled using [`unsync`](@ref). This can always be
done for the last such loop in a `@threaded` expression, since there are no
subsequent loops that require consistency of shared memory reads. Every other
loop over a `@shmem_threaded` iterator that does not write data to shared memory
arrays should also be modified in this manner, since unnecessary synchronization
pauses can degrade performance.

Since each GPU thread processes a subset of the elements in an iterator
annotated with `@shmem_threaded`, functions that are meant to process the entire
iterator, like `sum` and `minimum`, would not generate valid results for its
subsets. If functions like this need to be evaluated, their results should
always be precomputed before `@threaded` loops.

When the `size` or `length` of an iterator annotated with `@shmem_threaded` is
used to compute the dimensions of a [`shmem_array`](@ref), it must be inferrable
during compilation to avoid unnecessary runtime allocations. If this is not
already the case, [`static_resize`](@ref) can be used to assign the iterator a
static size.
"""
macro shmem_threaded end

"""
    threaded(f, device, itrs...; [shmem_threaded], kwargs...)

Functional form of [`@threaded`](@ref). Depending on whether the
`@shmem_threaded` iterator is specified, this is equivalent to one of the
following expressions when `n` other iterators are provided:

```julia
@threaded device kwargs... begin
    for item_n in itrs[n], ..., item_2 in itrs[2], item_1 in itrs[1]
        f(item_1, item_2, ..., item_n)
    end
end
```

```julia
@threaded device kwargs... begin
    @shmem_threaded shmem_itr = shmem_threaded
    for item_n in itrs[n], ..., item_2 in itrs[2], item_1 in itrs[1]
        f(item_1, item_2, ..., item_n, shmem_itr)
    end
end
```

If only the `shmem_threaded` iterator is available, this is instead equivalent
to the following expression:

```julia
@threaded device kwargs... begin
    @shmem_threaded shmem_itr = shmem_threaded
    begin
        f(shmem_itr)
    end
end
```

On single-threaded CPU devices, `@threaded` and `@shmem_threaded` annotations
are eliminated without any intermediate function calls, so the expressions with
macros will typically compile slightly faster than the `threaded` function. On
other devices, the only differences between the macro expressions and the
`threaded` function are syntactic:
 - the body of the `@threaded` loop is specified as a function,
 - the `@shmem_threaded` iterator is specified as a keyword argument,
 - every symbol used as a keyword argument is wrapped in a `Val`, e.g., `:auto`
   is specified as `Val(:auto)`.
Although the macro expressions are internally converted into `threaded` function
calls, these differences make the functional form somewhat harder to read, so it
should generally be avoided.
"""
function threaded(
    f::F,
    device,
    itr;
    shmem_threaded = nothing,
    coarsen = Val(:dynamic),
    shmem_coarsen = Val(:auto),
    block_size = Val(:auto),
    verbose = false,
) where {F}
    (coarsen isa Val || coarsen isa Integer && coarsen > 0) ||
        throw(ArgumentError("`coarsen` is not positive"))
    (shmem_coarsen isa Val || shmem_coarsen isa Integer && shmem_coarsen > 0) ||
        throw(ArgumentError("`shmem_coarsen` is not positive"))
    (block_size isa Val || block_size isa Integer && block_size > 0) ||
        throw(ArgumentError("`block_size` is not positive"))
    verbose isa Bool || throw(ArgumentError("`verbose` is not a Bool"))
    threadable_itr = threadable(device, itr)
    isempty(threadable_itr) && return nothing
    threaded_on_device(
        f,
        device,
        threadable_itr,
        shmem_threaded;
        coarsen,
        shmem_coarsen,
        block_size,
        verbose,
    )
end

threaded(f::F, device, itrs...; kwargs...) where {F} =
    threaded(device, Iterators.product(itrs...); kwargs...) do items, args...
        f(items..., args...)
    end
