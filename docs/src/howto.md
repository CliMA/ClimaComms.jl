# How-to Guide

Task-oriented recipes for common `ClimaComms` uses. If you are new to
`ClimaComms`, start with [Getting Started](@ref).

## How to run a script on a GPU

Set the `CLIMACOMMS_DEVICE` environment variable to `CUDA`, either in your
shell (note: no spaces around `=`)

```bash
export CLIMACOMMS_DEVICE="CUDA"
```

or at the top of your Julia script, before the device is constructed:

```julia
ENV["CLIMACOMMS_DEVICE"] = "CUDA"
```

`CUDA.jl` must be installed in your environment and loaded before the
device is used; adding [`ClimaComms.@import_required_backends`](@ref) at
the top of the script (after `import ClimaComms`) takes care of loading.

## How to run a script with MPI

Set `CLIMACOMMS_CONTEXT` to `MPI` and launch Julia with an MPI launcher:

```bash
CLIMACOMMS_CONTEXT=MPI mpiexec -n 4 julia --project script.jl
```

`MPI.jl` must be installed in your environment; use
[`ClimaComms.@import_required_backends`](@ref) to load it. Call
[`ClimaComms.init`](@ref) on the context before performing any
communication; it initializes MPI and, for GPU runs, assigns a GPU to each
rank on a node.

## How to allocate arrays on the right device

Use [`ClimaComms.array_type`](@ref), which returns `Array` for CPU devices
and `CuArray` for `CUDADevice`:

```julia
ArrayType = ClimaComms.array_type(ClimaComms.device())
x = ArrayType([1.0, 2.0, 3.0])
zeros_on_device = ArrayType(zeros(100))
```

## How to write functions that specialize on the device

Devices are empty structs, so you can dispatch on them like any other
type. This is how low-level CliMA code provides CPU and GPU
implementations of the same operation:

```julia
import ClimaComms: AbstractCPUDevice, CUDADevice
import CUDA

my_allocate(::AbstractCPUDevice, data) = Array(data)
my_allocate(::CUDADevice, data) = CUDA.CuArray(data)
```

Higher-level code can then call `my_allocate(device, data)` without knowing
what hardware it runs on.

## How to parallelize a loop on any device

Use [`ClimaComms.@threaded`](@ref), a device-flexible generalization of
`Threads.@threads`. The same loop runs serially on a
`CPUSingleThreaded` device, across threads on a `CPUMultiThreaded` device,
and as a CUDA kernel on a `CUDADevice`:

```julia
function threaded_add!(a, b, device)
    ClimaComms.@threaded device for i in eachindex(a, b)
        a[i] += b[i]
    end
end
```

The macro accepts options for performance tuning:

- `coarsen` controls how many loop iterations each thread evaluates,
  reducing the overhead of launching threads;
- `block_size` controls the number of threads per block on GPUs.

Lazy iterators (`zip`, `enumerate`, `Iterators.product`, generator
expressions) and multiple loop variables are supported. See the
[`ClimaComms.@threaded`](@ref) docstring for details and for the
type-inference requirements of GPU execution.

!!! note
    On GPUs, all values used in the loop body must have statically
    inferrable types. In particular, wrap `@threaded` loops in functions
    rather than executing them at global scope.

## How to time and synchronize device code

GPU kernels launch asynchronously, so `Base.@time` measures only the launch
cost. Use the device-flexible macros, which fall back to the `Base`
versions on CPUs and use the `CUDA.jl` versions on GPUs:

```julia
device = ClimaComms.device()

ClimaComms.@time device my_kernel!(x)         # @time / CUDA.@time
dt = ClimaComms.@elapsed device my_kernel!(x) # @elapsed / CUDA.@elapsed
ClimaComms.@sync device my_kernel!(x)         # @sync / CUDA.@sync
```

Use [`ClimaComms.@cuda_sync`](@ref) when you only need to synchronize GPU
work (it is a no-op on CPUs, avoiding the overhead of `Base.@sync`).

## How to print or log only from the root process

Guard output with [`ClimaComms.iamroot`](@ref):

```julia
ClimaComms.iamroot(context) && @info "This prints once, not once per rank"
```

Or install a logger that silences non-root processes globally:

```julia
using Logging
Logging.global_logger(ClimaComms.OnlyRootLogger(context))
```

See [Logging](@ref) for per-rank log files ([`ClimaComms.FileLogger`](@ref))
and rank-prefixed messages ([`ClimaComms.MPILogger`](@ref)).

## How to query available device memory

[`ClimaComms.free_memory`](@ref) and [`ClimaComms.total_memory`](@ref)
report bytes of memory on the device (system memory for CPU devices, GPU
memory for `CUDADevice`):

```julia
device = ClimaComms.device()
frac_free = ClimaComms.free_memory(device) / ClimaComms.total_memory(device)
```

## How to inspect the current configuration

Use `Base.summary` on a context to print the context type, device, and (for
MPI runs) the rank layout across nodes:

```julia
summary(stdout, context)
```

To verify that MPI and CUDA modules are set up correctly on a cluster, see
[this guide](https://github.com/CliMA/slurm-buildkite?tab=readme-ov-file#testing-cuda-and-mpi-modules).
