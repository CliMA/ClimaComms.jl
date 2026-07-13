# Getting Started

This tutorial installs `ClimaComms` and builds up a small script that runs
unchanged on a single CPU thread, on a GPU, and across MPI processes. At the
end, you will know how to select devices and contexts, allocate arrays on
the right device, and communicate between processes.

## Installation

`ClimaComms` is registered in the Julia General registry:

```julia
using Pkg
Pkg.add("ClimaComms")
```

To run on GPUs or with MPI, also install the corresponding backend packages
(`ClimaComms` loads them through package extensions, so they are not hard
dependencies):

```julia
Pkg.add("CUDA")   # for NVIDIA GPUs
Pkg.add("MPI")    # for distributed runs
```

## A first script

Save the following as `script.jl`:

```julia
import ClimaComms

# Load MPI.jl and/or CUDA.jl if the environment variables request them.
ClimaComms.@import_required_backends

# Select the context and device from the environment variables.
context = ClimaComms.context()
device = ClimaComms.device(context)

# Initialize the context (e.g., set up MPI and assign GPUs to ranks).
# Returns this process's ID (1-based) and the total number of processes.
mypid, nprocs = ClimaComms.init(context)

# Allocate an array on the selected device (Array on CPUs, CuArray on GPUs).
ArrayType = ClimaComms.array_type(device)
my_array = mypid * ArrayType([1.0, 1.0, 1.0])

# Sum the arrays from all processes; the result is valid on the root process.
reduced_array = ClimaComms.reduce(context, my_array, +)
ClimaComms.iamroot(context) && @show reduced_array
```

Walking through the pieces:

- [`ClimaComms.@import_required_backends`](@ref) imports `MPI.jl` and/or
  `CUDA.jl` when the environment variables request them, so you do not have
  to edit the script to switch backends. The packages must be installed in
  your environment. (Only use this macro in scripts, never in library code.)
- [`ClimaComms.context`](@ref) reads `CLIMACOMMS_CONTEXT` and returns a
  [`SingletonCommsContext`](@ref ClimaComms.SingletonCommsContext) (the
  default) or an [`MPICommsContext`](@ref ClimaComms.MPICommsContext).
  [`ClimaComms.device`](@ref) similarly reads `CLIMACOMMS_DEVICE`.
- [`ClimaComms.init`](@ref) performs any backend initialization and returns
  the process ID and process count. In single-process runs it returns
  `(1, 1)`.
- [`ClimaComms.array_type`](@ref) returns the array type matching the
  device, so allocations land on the right hardware.
- [`ClimaComms.reduce`](@ref) combines values across processes. In
  single-process runs, it (like all communication primitives) is a no-op.
- [`ClimaComms.iamroot`](@ref) is `true` only on the root process, so
  output is printed once rather than once per process.

## Running the script everywhere

On a single CPU thread (the default):

```bash
julia --project script.jl
```

On a GPU (requires `CUDA.jl` and a CUDA-capable GPU):

```bash
CLIMACOMMS_DEVICE=CUDA julia --project script.jl
```

On four MPI processes (requires `MPI.jl`):

```bash
CLIMACOMMS_CONTEXT=MPI mpiexec -n 4 julia --project script.jl
```

On four MPI processes, each driving its own GPU:

```bash
CLIMACOMMS_DEVICE=CUDA CLIMACOMMS_CONTEXT=MPI mpiexec -n 4 julia --project script.jl
```

The script itself never changes — this is the central promise of
`ClimaComms`.

## Next steps

- The [How-to Guide](@ref) collects recipes for common tasks, such as
  writing loops that parallelize on any device.
- The [Design Philosophy](@ref) page explains how `ClimaComms` is designed
  and how the CliMA packages build on it.
