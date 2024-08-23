# ClimaComms

`ClimaComms.jl` is a small package that provides abstractions for different
computing devices and environments. `ClimaComms.jl` is use extensively by
`CliMA` packages to control where and how simulations are run (e.g., one on
core, on multiple GPUs, et cetera).

This page highlights the most important user-facing `ClimaComms` concepts. If
you are using `ClimaComms` as a developer, refer to the [Developing with
`ClimaComms`](@ref) page. For a detailed list of all the functions and objects
implemented, the [APIs](@ref) page collects all of them.

## `Device`s and `Context`s

The two most important objects in `ClimaComms.jl` are the [`Device`](@ref
`AbstractDevice`) and the [`Context`](@ref `AbstractCommsContext`).

A `Device` identifies a computing device, a piece of hardware that will be
executing some code. The `Device`s currently implemented are
- [`CPUSingleThreaded`](@ref), for a CPU core with a single thread;
- [`CUDADevice`](@ref), for a single CPU core.

> :warning: [`CPUMultiThreaded`](@ref) is also available, but this device is not
> actively used or developed.

`Device`s are part of [`Context`](@ref `AbstractCommsContext`)s,
objects that contain information require for multiple `Device`s to communicate.
Implemented `Context`s are
- [`SingletonCommsContext`](@ref), when there is no parallelism;
- [`MPICommsContext`](@ref), for a MPI-parallelized runs.

To choose a device and a context, most `CliMA` packages use the
[`device()`](@ref) and [`context()`](@ref) functions. These functions look at
specific environment variables and set the `device` and `context` accordingly.
By default, the [`CPUSingleThreaded`](@ref) device is chosen and the context is
set to [`SingletonCommsContext`](@ref) unless `ClimaComms` detects being run in
a standard MPI launcher (as `srun` or `mpiexec`).

For example, to run a simulation on a GPU, run `julia` as
```bash
export CLIMACOMMS_DEVICE="CUDA"
export CLIMACOMMS_CONTEXT="SINGLETON"
# call/open julia as usual
```

> Note: there might be other ways to control the device and context. Please,
> refer to the documentation of the specific package to learn more.

## Running with MPI/CUDA

`CliMA` packages do not depend directly on `MPI` or `CUDA`, so, if you want to
run your simulation in parallel mode and/or on GPUs, you will need to install
some packages separately.

For parallel simulations, [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl), and
for GPU runs, [`CUDA.jl`](https://github.com/JuliaGPU/CUDA.jl). You can install
these packages in your base environment
```bash
julia -E "using Pkg; Pkg.add(\"CUDA\"); Pkg.add(\"MPI\")"
```
Some packages come with environments that includes all possible backends
(typically `.buildkite`). You can also consider directly using those
environments.
