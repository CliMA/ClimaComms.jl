import ..ClimaComms

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

The default is `CPU`.
"""
function device()
    target_device = device_type()
    if target_device == :CUDADevice && !cuda_ext_is_loaded()
        error(
            "Loading CUDA.jl is required to use CUDADevice. You might want to call ClimaComms.@import_required_backends",
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
    LocalArray{T, S}(device)

A device-flexible local array whose element type `T` and size `S` are known
during compilation, which corresponds to an `MArray` on CPUs and a
`CuStaticSharedArray` on NVIDIA GPUs. A `LocalArray` is typically much faster to
read and modify than generic arrays like `Array` or `CuArray`.
"""
abstract type LocalArray{T, S} end
LocalArray{T, S}(::AbstractCPUDevice) where {T, S} =
    MArray{Tuple{S...}, T}(undef)

"""
    LocalVector{T, N}(device)

Alias for a one-dimensional `LocalArray{T, (N,)}`.
"""
abstract type LocalVector{T, N} end
LocalVector{T, N}(device) where {T, N} = LocalArray{T, (N,)}(device)

"""
    local_sync(device)

Internal function that synchronizes all threads with access to the same
`LocalArray`s.
"""
local_sync(::AbstractCPUDevice) = nothing

"""
    @threaded [device] [coarsen=...] [block_size=...] for <item> in <itr> ... end

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
macro threaded end

"""
    @threaded [device] [coarsen=...] [block_size=...] for <item> in <itr>
        ...
        @gpu_threaded for <gpu_threaded_item> in <gpu_threaded_itr> ... end
        ...
    end

The `@gpu_threaded` macro can be used inside an `@threaded` loop to implement
a second level of parallelization on GPUs. Using `@gpu_threaded` changes the
behavior of `@threaded` so that the outer loop is only parallelized across GPU
blocks, while the inner loop is parallelized across GPU threads. It also allows
the `coarsen` argument to be a `Tuple` of two integers, where the the first
represents coarsening of the outer loop's iterator and the second represents
coarsening of the inner loop's iterator. (When only one integer is used, the
inner loop's iterator gets a default coarsening value of 1.) Every
`@gpu_threaded` loop in a single `@threaded` loop must have the same iterator
and variable.

On CPUs, `@gpu_threaded` has no effect, and it can be omitted without impacting
performance or latency. On GPUs, it can be used conjunction with `LocalArray`s
or `LocalVector`s to synchronize shared memory across GPU threads.
"""
macro gpu_threaded(_...)
    throw(ArgumentError("@gpu_threaded can only be inside an @threaded loop"))
end

macro threaded(args...)
    usage_string = "Usage: @threaded [device] [coarsen=...] [block_size=...] for ... end"
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

    loop_expr = args[n_args]
    unthreaded_expr = escaped_f_expr(loop_expr)
    function_defs_expr, itr_exprs =
        escaped_function_defs_and_itr_exprs(loop_expr)

    return quote
        $function_defs_expr
        device = $(esc(device_expr))
        if device isa CPUSingleThreaded
            $unthreaded_expr
        else
            coarsen = $(esc(coarsen_expr))
            block_size = $(esc(block_size_expr))
            if coarsen isa Val
                threaded(f, device, coarsen, $(itr_exprs...); block_size)
            else
                threaded(coarse_f, device, coarsen, $(itr_exprs...); block_size)
            end
        end
    end
end

function escaped_function_defs_and_itr_exprs(threaded_arg_expr)
    Meta.isexpr(threaded_arg_expr, :for) ||
        throw(ArgumentError("argument of @threaded must be a for-loop"))
    (var_and_itr_expr, loop_iteration_expr) = threaded_arg_expr.args
    Meta.isexpr(var_and_itr_expr, :(=)) ||
        throw(ArgumentError("@threaded loop can only use one iterator"))
    (var_expr, itr_expr) = var_and_itr_expr.args
    (gpu_threaded_var_expr, gpu_threaded_itr_expr) =
        gpu_threaded_var_and_itr_exprs(loop_iteration_expr)
    f_expr = escaped_f_expr(loop_iteration_expr)
    coarse_f_expr = quote
        for itr_index in itr_indices
            @inbounds $(esc(var_expr)) = $(esc(itr_expr))[itr_index]
            $f_expr
        end
    end
    f_def_expr = :(f($(esc(var_expr))) = $f_expr)
    coarse_f_def_expr = :(coarse_f(itr_indices) = $coarse_f_expr)
    function_defs_expr_args = [f_def_expr, coarse_f_def_expr]
    if gpu_threaded_var_expr != nothing
        gpu_threaded_f_expr = escaped_gpu_threaded_f_expr(loop_iteration_expr)
        gpu_threaded_coarse_f_loop_iteration_expr =
            escaped_gpu_threaded_coarse_f_expr(
                loop_iteration_expr,
                gpu_threaded_var_expr,
                gpu_threaded_itr_expr,
            )
        gpu_threaded_coarse_f_expr = quote
            for itr_index in itr_indices
                @inbounds $(esc(var_expr)) = $(esc(itr_expr))[itr_index]
                $(gpu_threaded_coarse_f_loop_iteration_expr)
            end
        end
        gpu_threaded_f_def_expr = :(
            f($(esc(var_expr)), $(esc(gpu_threaded_var_expr))) =
                $gpu_threaded_f_expr
        )
        gpu_threaded_coarse_f_def_expr = :(
            coarse_f(itr_indices, gpu_threaded_itr_indices) =
                $gpu_threaded_coarse_f_expr
        )
        append!(
            function_defs_expr_args,
            [gpu_threaded_f_def_expr, gpu_threaded_coarse_f_def_expr],
        )
    end
    itr_exprs =
        gpu_threaded_var_expr == nothing ? [esc(itr_expr)] :
        [esc(itr_expr), esc(gpu_threaded_itr_expr)]
    return Expr(:block, function_defs_expr_args...), itr_exprs
end

is_gpu_threaded_macro_call_expr(arg) =
    Meta.isexpr(arg, :macrocall) &&
    arg.args[1] in (:(var"@gpu_threaded"), :(ClimaComms.var"@gpu_threaded"))

gpu_threaded_var_and_itr_exprs(arg) =
    if is_gpu_threaded_macro_call_expr(arg)
        gpu_threaded_arg_expr = arg.args[3]
        Meta.isexpr(gpu_threaded_arg_expr, :for) ||
            throw(ArgumentError("argument of @gpu_threaded must be a for-loop"))
        var_and_itr_expr = gpu_threaded_arg_expr.args[1]
        Meta.isexpr(var_and_itr_expr, :(=)) ||
            throw(ArgumentError("@gpu_threaded loop can only use one iterator"))
        var_and_itr_expr.args
    elseif arg isa Expr
        var_and_itr_exprs = Any[nothing, nothing]
        for inner_arg in arg.args
            new_var_and_itr_exprs = gpu_threaded_var_and_itr_exprs(inner_arg)
            new_var_and_itr_exprs == Any[nothing, nothing] && continue
            if var_and_itr_exprs == Any[nothing, nothing]
                var_and_itr_exprs = new_var_and_itr_exprs
            elseif var_and_itr_exprs != new_var_and_itr_exprs
                throw(ArgumentError("all @gpu_threaded loops in a @threaded \
                                     loop must use the same iterator and \
                                     variable name"))
            end
        end
        var_and_itr_exprs
    else
        Any[nothing, nothing]
    end

escaped_f_expr(arg) =
    if is_gpu_threaded_macro_call_expr(arg)
        esc(arg.args[3])
    elseif arg isa Expr
        Expr(arg.head, Base.mapany(escaped_f_expr, arg.args)...)
    else
        arg isa Symbol ? esc(arg) : arg
    end

escaped_gpu_threaded_f_expr(arg) =
    if is_gpu_threaded_macro_call_expr(arg)
        gpu_threaded_arg_expr = arg.args[3]
        gpu_threaded_loop_iteration_expr = gpu_threaded_arg_expr.args[2]
        quote
            $(esc(gpu_threaded_loop_iteration_expr))
            local_sync(device)
        end
    elseif arg isa Expr
        Expr(arg.head, Base.mapany(escaped_gpu_threaded_f_expr, arg.args)...)
    else
        arg isa Symbol ? esc(arg) : arg
    end

escaped_gpu_threaded_coarse_f_expr(
    arg,
    gpu_threaded_var_expr,
    gpu_threaded_itr_expr,
) =
    if is_gpu_threaded_macro_call_expr(arg)
        gpu_threaded_arg_expr = arg.args[3]
        gpu_threaded_loop_iteration_expr = gpu_threaded_arg_expr.args[2]
        quote
            for gpu_threaded_itr_index in gpu_threaded_itr_indices
                @inbounds $(esc(gpu_threaded_var_expr)) =
                    $(esc(gpu_threaded_itr_expr))[gpu_threaded_itr_index]
                $(esc(gpu_threaded_loop_iteration_expr))
            end
            local_sync(device)
        end
    elseif arg isa Expr
        new_inner_args = Base.mapany(arg.args) do inner_arg
            escaped_gpu_threaded_coarse_f_expr(
                inner_arg,
                gpu_threaded_var_expr,
                gpu_threaded_itr_expr,
            )
        end
        Expr(arg.head, new_inner_args...)
    else
        arg isa Symbol ? esc(arg) : arg
    end

# Ignore the block_size keyword argument when using @threaded on CPUs.
threaded(f_or_coarse_f::F, args...; block_size) where {F} =
    threaded(f_or_coarse_f, args...)

# Ignore @gpu_threaded iterators and their corresponding values of coarsen when
# using @threaded on CPUs.
threaded(f_or_coarse_f::F, device, coarsen, itr, _) where {F} =
    threaded(f_or_coarse_f, device, coarsen, itr)
threaded(f_or_coarse_f::F, device, coarsen::Tuple{Int, Int}, itr, _) where {F} =
    threaded(f_or_coarse_f, device, coarsen[1], itr)

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

function threaded(
    coarse_f::F,
    ::CPUMultiThreaded,
    items_in_thread::Int,
    itr,
) where {F}
    items_in_thread > 0 ||
        throw(ArgumentError("integer `coarsen` value must be positive"))
    Base.require_one_based_indexing(itr)

    threads = cld(length(itr), items_in_thread)
    Threads.@threads :static for thread_index in 1:threads
        first_item_index = items_in_thread * (thread_index - 1) + 1
        last_item_index = items_in_thread * thread_index
        coarse_f(first_item_index:min(last_item_index, length(itr)))
    end
end
