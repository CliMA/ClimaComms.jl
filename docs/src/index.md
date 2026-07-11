# ClimaComms.jl

`ClimaComms.jl` provides the abstractions for computing devices and
communication contexts on which the [CliMA](https://github.com/CliMA)
ecosystem is built. It lets the same simulation code run unchanged on a
single CPU thread, on multiple CPU threads, on NVIDIA GPUs, and across many
nodes with MPI: the device and the parallelism are selected at runtime,
typically through environment variables.

## The two central abstractions

A **device** ([`ClimaComms.AbstractDevice`](@ref)) identifies the hardware
that executes code. The devices currently implemented are

- [`CPUSingleThreaded`](@ref ClimaComms.CPUSingleThreaded): a CPU using a single thread,
- [`CPUMultiThreaded`](@ref ClimaComms.CPUMultiThreaded): a CPU using multiple threads,
- [`CUDADevice`](@ref ClimaComms.CUDADevice): a single CUDA-enabled GPU.

A **context** ([`ClimaComms.AbstractCommsContext`](@ref)) is the environment
through which processes communicate. It wraps a device and, for distributed
runs, the information needed to exchange data between processes. The
contexts currently implemented are

- [`SingletonCommsContext`](@ref ClimaComms.SingletonCommsContext): a single process, no parallelism;
- [`MPICommsContext`](@ref ClimaComms.MPICommsContext): distributed runs via MPI.

Devices and contexts are selected at runtime with the
[`ClimaComms.device`](@ref) and [`ClimaComms.context`](@ref) functions, which
read the `CLIMACOMMS_DEVICE` and `CLIMACOMMS_CONTEXT` environment variables.
For example, to run a script on a GPU with four MPI processes:

```bash
export CLIMACOMMS_DEVICE="CUDA"
export CLIMACOMMS_CONTEXT="MPI"
mpiexec -n 4 julia --project script.jl
```

!!! note
    Some packages provide additional ways to control the device and context
    (e.g., configuration files). Refer to the documentation of the specific
    package to learn more.

## Where to go next

- [Getting Started](@ref): install `ClimaComms` and write a first script
  that runs on any device and any number of processes.
- [How-to Guide](@ref): recipes for common tasks, such as running on GPUs,
  writing device-agnostic loops, and setting up logging for MPI runs.
- [Design Philosophy](@ref): why `ClimaComms` exists, how it is designed,
  and how the CliMA packages use it.
- [Logging](@ref): loggers for distributed runs.
- [Frequently Asked Questions](@ref): solutions to common problems.
- [APIs](@ref): the complete API reference.
