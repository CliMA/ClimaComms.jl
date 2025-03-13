import ..ClimaComms
import StaticArrays: MArray

"""
    AbstractDevice

The base type for a device.
"""
abstract type AbstractDevice end

"""
    AbstractCPUDevice()

Abstract device type for single-threaded and multi-threaded CPU runs.
"""
abstract type AbstractCPUDevice <: AbstractDevice end


"""
    CPUSingleThreaded()

Use the CPU with single thread.
"""
struct CPUSingleThreaded <: AbstractCPUDevice end

"""
    CPUMultiThreaded()

Use the CPU with multiple thread.
"""
struct CPUMultiThreaded <: AbstractCPUDevice end

"""
    CUDADevice()

Use NVIDIA GPU accelarator
"""
struct CUDADevice <: AbstractDevice end

"""
    MetalDevice()

Use Apple Metal GPU accelerator for M-series chips.
"""
struct MetalDevice <: AbstractDevice end

"""
    ClimaComms.device_functional(device)

Return true when the `device` is correctly set up.
"""
function device_functional end

device_functional(::CPUSingleThreaded) = true
device_functional(::CPUMultiThreaded) = true

function device_type()
    env_var = get(ENV, "CLIMACOMMS_DEVICE", "CPU")
    if env_var == "CPU"
        return Threads.nthreads() > 1 ? :CPUMultiThreaded : :CPUSingleThreaded
    elseif env_var == "CPUSingleThreaded"
        return :CPUSingleThreaded
    elseif env_var == "CPUMultiThreaded"
        return :CPUMultiThreaded
    elseif env_var == "CUDA"
        return :CUDADevice
    elseif env_var == "Metal"
        return :MetalDevice
    else
        error("Invalid CLIMACOMMS_DEVICE: $env_var")
    end
end

"""
    ClimaComms.device()

Determine the device to use depending on the `CLIMACOMMS_DEVICE` environment variable.

Allowed values:
- `CPU`, single-threaded or multi-threaded depending on the number of threads;
- `CPUSingleThreaded`,
- `CPUMultiThreaded`,
- `CUDA`.
- `Metal`.

The default is `CPU`.
"""
function device()
    target_device = device_type()
    if target_device == :CUDADevice && !cuda_ext_is_loaded()
        error(
            "Loading CUDA.jl is required to use CUDADevice. You might want to call ClimaComms.@import_required_backends",
        )
    elseif target_device == :MetalDevice && !metal_ext_is_loaded()
        error(
            "Loading Metal.jl is required to use MetalDevice. You might want to call ClimaComms.@import_required_backends",
        )
    end
    DeviceConstructor = getproperty(ClimaComms, target_device)
    return DeviceConstructor()
end

Base.summary(io::IO, device::AbstractDevice) = string(device_type())

"""
    ClimaComms.array_type(::AbstractDevice)

The base array type used by the specified device (currently `Array` or `CuArray`).
"""
array_type(::AbstractCPUDevice) = Array

"""
Internal function that can be used to assign a device to a process.

Currently used to assign CUDADevices to MPI ranks.
"""
_assign_device(device, id) = nothing

"""
    @time f(args...; kwargs...)

Device-flexible `@time`:

Calls
```julia
@time f(args...; kwargs...)
```
for CPU devices and
```julia
CUDA.@time f(args...; kwargs...)
```
for CUDA devices.
"""
function time(f::F, device::AbstractCPUDevice, args...; kwargs...) where {F}
    Base.@time begin
        f(args...; kwargs...)
    end
end

"""
    elapsed(f::F, device::AbstractDevice, args...; kwargs...)

Device-flexible `elapsed`.

Calls
```julia
@elapsed f(args...; kwargs...)
```
for CPU devices and
```julia
CUDA.@elapsed f(args...; kwargs...)
```
for CUDA devices.
"""
function elapsed(f::F, device::AbstractCPUDevice, args...; kwargs...) where {F}
    Base.@elapsed begin
        f(args...; kwargs...)
    end
end

"""
    sync(f, ::AbstractDevice, args...; kwargs...)

Device-flexible function that calls `@sync`.

Calls
```julia
@sync f(args...; kwargs...)
```
for CPU devices and
```julia
CUDA.@sync f(args...; kwargs...)
```
for CUDA devices.

An example use-case of this might be:
```julia
BenchmarkTools.@benchmark begin
    if ClimaComms.device() isa ClimaComms.CUDADevice
        CUDA.@sync begin
            launch_cuda_kernels_or_spawn_tasks!(...)
        end
    elseif ClimaComms.device() isa ClimaComms.CPUMultiThreading
        Base.@sync begin
            launch_cuda_kernels_or_spawn_tasks!(...)
        end
    end
end
```

If the CPU version of the above example does not leverage
spawned tasks (which require using `Base.sync` or `Threads.wait`
to synchronize), then you may want to simply use [`cuda_sync`](@ref).
"""
function sync(f::F, ::AbstractCPUDevice, args...; kwargs...) where {F}
    Base.@sync begin
        f(args...; kwargs...)
    end
end

"""
    cuda_sync(f, ::AbstractDevice, args...; kwargs...)

Device-flexible function that (may) call `CUDA.@sync`.

Calls
```julia
f(args...; kwargs...)
```
for CPU devices and
```julia
CUDA.@sync f(args...; kwargs...)
```
for CUDA devices.
"""
function cuda_sync(f::F, ::AbstractCPUDevice, args...; kwargs...) where {F}
    f(args...; kwargs...)
end

"""
    allowscalar(f, ::AbstractDevice, args...; kwargs...)

Device-flexible version of `CUDA.@allowscalar`.

Lowers to
```julia
f(args...)
```
for CPU devices and
```julia
CUDA.@allowscalar f(args...)
```
for CUDA devices.

This is usefully written with closures via
```julia
allowscalar(device) do
    f()
end
```
"""
allowscalar(f, ::AbstractCPUDevice, args...; kwargs...) = f(args...; kwargs...)

"""
    @time device expr

Device-flexible `@time`.

Lowers to
```julia
@time expr
```
for CPU devices and
```julia
CUDA.@time expr
```
for CUDA devices.
"""
macro time(device, expr)
    __CC__ = ClimaComms
    return :($__CC__.time(() -> $(esc(expr)), $(esc(device))))
end

"""
    @sync device expr

Device-flexible `@sync`.

Lowers to
```julia
@sync expr
```
for CPU devices and
```julia
CUDA.@sync expr
```
for CUDA devices.

An example use-case of this might be:
```julia
BenchmarkTools.@benchmark begin
    if ClimaComms.device() isa ClimaComms.CUDADevice
        CUDA.@sync begin
            launch_cuda_kernels_or_spawn_tasks!(...)
        end
    elseif ClimaComms.device() isa ClimaComms.CPUMultiThreading
        Base.@sync begin
            launch_cuda_kernels_or_spawn_tasks!(...)
        end
    end
end
```

If the CPU version of the above example does not leverage
spawned tasks (which require using `Base.sync` or `Threads.wait`
to synchronize), then you may want to simply use [`@cuda_sync`](@ref).
"""
macro sync(device, expr)
    __CC__ = ClimaComms
    return :($__CC__.sync(() -> $(esc(expr)), $(esc(device))))
end

"""
    @cuda_sync device expr

Device-flexible `CUDA.@sync`.

Lowers to
```julia
expr
```
for CPU devices and
```julia
CUDA.@sync expr
```
for CUDA devices.
"""
macro cuda_sync(device, expr)
    __CC__ = ClimaComms
    return :($__CC__.cuda_sync(() -> $(esc(expr)), $(esc(device))))
end

"""
    @elapsed device expr

Device-flexible `@elapsed`.

Lowers to
```julia
@elapsed expr
```
for CPU devices and
```julia
CUDA.@elapsed expr
```
for CUDA devices.
"""
macro elapsed(device, expr)
    __CC__ = ClimaComms
    return :($__CC__.elapsed(() -> $(esc(expr)), $(esc(device))))
end

"""
    @assert device cond [text]

Device-flexible `@assert`.

Lowers to
```julia
@assert cond [text]
```
for CPU devices and
```julia
CUDA.@cuassert cond [text]
```
for CUDA devices.
"""
macro assert(device, cond, text = nothing)
    text_func = isnothing(text) ? nothing : :(() -> $(esc(text)))
    return :($assert($(esc(device)), () -> $(esc(cond)), $text_func))
end
assert(::AbstractCPUDevice, cond::C, text::T) where {C, T} =
    isnothing(text) ? (Base.@assert cond()) : (Base.@assert cond() text())

"""
    @threaded [device] [coarsen=...] [block_size=...] for ... end

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

To fix other kinds of inference issues on GPUs, especially ones brought about by
indexing into iterators with nonuniform element types, see
[`UnrolledUtilities.jl`](https://github.com/CliMA/UnrolledUtilities.jl).
"""
macro threaded(args...)
    usage_string = "Usage: @threaded [device] [coarsen=...] [block_size=...] \
                    for ... end"
    n_args = length(args)
    1 <= n_args <= 3 || throw(ArgumentError(usage_string))

    device_expr = :($ClimaComms.device())
    coarsen_expr = :(:dynamic)
    block_size_expr = :(:auto)
    for arg_number in 1:(n_args - 1)
        device_or_coarsen_expr = args[arg_number]
        if Meta.isexpr(device_or_coarsen_expr, :(=))
            (kwarg_name, kwarg_value) = device_or_coarsen_expr.args
            if kwarg_name == :coarsen
                coarsen_expr = kwarg_value
            elseif kwarg_name == :block_size
                block_size_expr = kwarg_value
            else
                throw(ArgumentError(usage_string))
            end
        else
            arg_number == 1 || throw(ArgumentError(usage_string))
            device_expr = device_or_coarsen_expr
        end
    end
    if coarsen_expr isa QuoteNode
        coarsen_expr = :(Val($(coarsen_expr)))
    end
    if block_size_expr isa QuoteNode
        block_size_expr = :(Val($(block_size_expr)))
    end

    loop_expr = if Meta.isexpr(args[n_args], :for)
        args[n_args]
    elseif (
        Meta.isexpr(args[n_args], :block) &&
        length(args[n_args].args) == 2 &&
        args[n_args].args[1] isa LineNumberNode &&
        Meta.isexpr(args[n_args].args[2], :for)
    )
        args[n_args].args[2]
    else
        throw(ArgumentError(usage_string))
    end
    (loop_def_expr, loop_body_expr) = loop_expr.args
    item_and_itr_exprs = if Meta.isexpr(loop_def_expr, :(=))
        [loop_def_expr.args]
    elseif Meta.isexpr(loop_def_expr, :block) && length(loop_def_expr.args) == 2
        [loop_def_expr.args[1].args, loop_def_expr.args[2].args]
    else
        throw(ArgumentError("@threaded only supports one or two loops"))
    end

    is_interdependent_itr_expr(expr) =
        Meta.isexpr(expr, :macrocall) && expr.args[1] in
        (Symbol("@interdependent"), :(ClimaComms.var"@interdependent"))
    interdependent_item_and_itr_exprs =
        filter(is_interdependent_itr_expr ∘ last, item_and_itr_exprs)
    independent_item_and_itr_exprs =
        filter(!is_interdependent_itr_expr ∘ last, item_and_itr_exprs)
    one_loop_string = "@threaded only supports up to one loop of the form "
    length(interdependent_item_and_itr_exprs) <= 1 ||
        throw(ArgumentError(one_loop_string * "`var in @interdependent(itr)`"))
    length(independent_item_and_itr_exprs) <= 1 ||
        throw(ArgumentError(one_loop_string * "`var in itr`"))

    if isempty(independent_item_and_itr_exprs)
        # If independent itr is missing, use (nothing,) as a default.
        independent_item_expr = :_
        independent_itr_expr = :((nothing,))
        coarsen_expr = :((Val(:dynamic), $coarsen_expr))
    else
        (independent_item_expr, independent_itr_expr) =
            independent_item_and_itr_exprs[1]
    end

    if isempty(interdependent_item_and_itr_exprs)
        return quote
            device = $(esc(device_expr))
            device isa CPUSingleThreaded ? $(esc(loop_expr)) :
            threaded(
                $(esc(independent_item_expr)) -> $(esc(loop_body_expr)),
                device,
                $(esc(coarsen_expr)),
                $(esc(independent_itr_expr));
                block_size = $(esc(block_size_expr)),
            )
        end
    else
        interdependent_item_expr = interdependent_item_and_itr_exprs[1][1]
        interdependent_itr_expr =
            interdependent_item_and_itr_exprs[1][2].args[3]

        is_sync_interdependent_expr(expr) =
            Meta.isexpr(expr, :macrocall) && expr.args[1] in (
                Symbol("@sync_interdependent"),
                :(ClimaComms.var"@sync_interdependent"),
            )
        autofill_sync_interdependent!(expr, item_expr) =
            if is_sync_interdependent_expr(expr)
                length(expr.args) == 3 && insert!(expr.args, 3, item_expr)
            elseif expr isa Expr
                for arg in expr.args
                    autofill_sync_interdependent!(arg, item_expr)
                end
            end
        autofill_sync_interdependent!(loop_body_expr, interdependent_item_expr)

        return quote
            device = $(esc(device_expr))
            device isa CPUSingleThreaded ?
            for $(esc(independent_item_expr)) in $(esc(independent_itr_expr))
                $(esc(interdependent_item_expr)) =
                    AllInterdependentItems($(esc(interdependent_itr_expr)))
                $(esc(loop_body_expr))
            end :
            threaded(
                device,
                $(esc(coarsen_expr)),
                $(esc(independent_itr_expr)),
                $(esc(interdependent_itr_expr));
                block_size = $(esc(block_size_expr)),
            ) do $(esc(independent_item_expr)), $(esc(interdependent_item_expr))
                $(esc(loop_body_expr))
            end
        end
    end
end

"""
    @interdependent(itr)

Annotation for an iterator of an `@threaded` loop that allows elements of the
iterator to affect each other within the loop. If an unannotated iterator is
used in an `@threaded` loop, its elements are assumed to be independent (naively
parallelizable). Items from an `@interdependent` iterator can only be used
within `@sync_interdependent` expressions, ensuring that all interdependencies
are isolated by thread synchronizations on GPUs. Since threads cannot be
efficiently synchronized on CPUs, all items in an `@interdependent` iterator are
processed on a single CPU thread, and each `@sync_interdependent` expression is
transformed into a separate for-loop.

This annotation can be used in conjunction with `static_shared_memory_array` to
implement performant kernels with interdependent threads in a device-agnostic
way, e.g.,

```julia-repl
julia> first_axis_derivative!(device, a, ::Val{N}) where {N} =
           ClimaComms.@threaded device for i in @interdependent(1:N), j in axes(a, 2)
               a_j = ClimaComms.static_shared_memory_array(device, eltype(a), N)
               ClimaComms.@sync_interdependent a_j[i] = a[i, j]
               ClimaComms.@sync_interdependent if i == 1
                   a[1, j] = a_j[2] - a_j[1]
               elseif i == N
                   a[N, j] = a_j[N] - a_j[N - 1]
               else
                   a[i, j] = (a_j[i + 1] - 2 * a_j[i] + a_j[i - 1]) / 2
               end
           end
first_axis_derivative! (generic function with 1 method)

julia> AT = ClimaComms.array_type(ClimaComms.device()); # Array or CuArray

julia> a = AT(rand(100, 1000));

julia> first_axis_derivative!(ClimaComms.device(), a, Val(100))
```

A `@threaded` loop can have at most one independent iterator and one
interdependent iterator. When two iterators are specified, as in the example
above, the independent iterator is parallelized across GPU blocks, and the
interdependent iterator is parallelized across threads within each block.
Passing a single integer value to `coarsen` will only apply thread coarsening to
the independent iterator when there are two iterators, but `coarsen` can also be
specified as a pair of integers, in which case the second integer controls
thread coarsening of the interdependent iterator.
"""
macro interdependent(itr)
    throw(ArgumentError("@interdependent can only be used to specify an \
                         iterator of a @threaded loop"))
end

# Ignore the block_size keyword argument on CPUs.
threaded(f::F, device::AbstractCPUDevice, args...; block_size) where {F} =
    threaded(f, device, args...)

# Ignore values of coarsen specified for interdependent iterators on CPUs.
threaded(
    f::F,
    device::AbstractCPUDevice,
    coarsen::NTuple{2, Any},
    independent_itr,
    interdependent_itr,
) where {F} =
    threaded(f, device, coarsen[1], independent_itr, interdependent_itr)

threaded(
    f::F,
    device::AbstractCPUDevice,
    coarsen,
    independent_itr,
    interdependent_itr,
) where {F} =
    threaded(device, coarsen, independent_itr) do independent_item
        f(independent_item, AllInterdependentItems(interdependent_itr))
    end

threaded(f::F, ::CPUMultiThreaded, ::Val{:dynamic}, itr) where {F} =
    Threads.@threads :dynamic for item in itr
        f(item)
    end

threaded(f::F, ::CPUMultiThreaded, ::Val{:static}, itr) where {F} =
    Threads.@threads :static for item in itr
        f(item)
    end

@static if VERSION >= v"1.11"
    threaded(f::F, ::CPUMultiThreaded, ::Val{:greedy}, itr) where {F} =
        Threads.@threads :greedy for item in itr
            f(item)
        end
end

function threaded(f::F, ::CPUMultiThreaded, items_in_thread::Int, itr) where {F}
    items_in_thread > 0 ||
        throw(ArgumentError("integer `coarsen` value must be positive"))
    Base.require_one_based_indexing(itr)

    threads = cld(length(itr), items_in_thread)
    Threads.@threads :static for thread_index in 1:threads
        first_item_index = items_in_thread * (thread_index - 1) + 1
        last_item_index = items_in_thread * thread_index
        for item_index in first_item_index:min(last_item_index, length(itr))
            @inbounds f(itr[item_index])
        end
    end
end

"""
    InterdependentIteratorData

Intermediate representation of an item from an `@interdependent` iterator.
Within each `@sync_interdependent` expression, this is replaced by a concrete
value and can be used as if it were in a standard for-loop.
"""
abstract type InterdependentIteratorData end
struct OneInterdependentItem{I, D} <: InterdependentIteratorData
    item::I
    device::D
end
struct MultipleInterdependentItems{T, I, D} <: InterdependentIteratorData
    itr::T
    indices::I
    device::D
end
struct AllInterdependentItems{T} <: InterdependentIteratorData
    itr::T
end

"""
    @sync_interdependent [item] code

Synchronizes a segment of code within an `@threaded` loop, allowing the
specified `item` from an `@interdependent` iterator to be used as if it were in
a standard for-loop. If this macro is used called from the same lexical scope as
its `@threaded` loop, the `item` variable can be determined automatically.

On GPU devices, the threads in every block are synchronized at the end of each
`@sync_interdependent` expression. In coarsened GPU threads, the code is
transformed into a for-loop over several items from the `@interdependent`
iterator, after which the threads in every block are synchronized. On CPU
devices, the code is transformed into a for-loop over the entire iterator.
"""
macro sync_interdependent(code_expr)
    throw(ArgumentError("Usage: @sync_interdependent item code"))
end
macro sync_interdependent(item_expr, code_expr)
    return quote
        sync_interdependent($(esc(item_expr))) do $(esc(item_expr))
            $(esc(code_expr))
        end
    end
end

function sync_interdependent(f::F, data::OneInterdependentItem) where {F}
    f(data.item)
    synchronize_gpu_threads(data.device)
end

function sync_interdependent(f::F, data::MultipleInterdependentItems) where {F}
    for index in data.indices
        @inbounds item = data.itr[index]
        f(item)
    end
    synchronize_gpu_threads(data.device)
end

sync_interdependent(f::F, data::AllInterdependentItems) where {F} =
    for item in data.itr
        f(item)
    end

"""
    synchronize_gpu_threads(device)

Synchronizes GPU threads with access to the same shared memory arrays. On CPU
devices, `synchronize_gpu_threads` does nothing. Every `@sync_interdependent`
expression is automatically followed by a call to this function on GPUs.
"""
synchronize_gpu_threads(::AbstractCPUDevice) = nothing

"""
    static_shared_memory_array(device, T, dims...)

Device-flexible array whose element type `T` and dimensions `dims` are known
during compilation, which corresponds to a static shared memory array on GPUs.
On CPUs, which do not provide access to high-performance shared memory, this
corresponds to an `MArray` instead. A `static_shared_memory_array` is much
faster to access and modify than generic arrays like `Array` or `CuArray`. In
`@threaded` loops, each `static_shared_memory_array` is shared by all threads
with the same independent iterator items.
"""
static_shared_memory_array(::AbstractCPUDevice, ::Type{T}, dims...) where {T} =
    MArray{Tuple{dims...}, T}(undef)
