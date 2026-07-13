# Design Philosophy

This page explains why `ClimaComms` exists, the design decisions behind it,
and how the CliMA packages build upon it. For hands-on instructions, see
[Getting Started](@ref) and the [How-to Guide](@ref).

## Why ClimaComms exists

Climate simulations run in radically different configurations: a scientist
debugs a column model on a laptop, tests a limited-area configuration on a
multi-core workstation or single GPU, and runs production simulations on many
GPUs or CPUs connected by MPI. Without an abstraction layer, every package would
have to write `if gpu ... elseif mpi ...` branches, and every simulation
script would be tied to one configuration.

`ClimaComms` centralizes this concern. It defines a small vocabulary — a
*device* for "what hardware executes the code" and a *context* for "how
processes communicate" — and provides device- and context-agnostic
operations on top of it. Code written against this vocabulary is portable
by construction: the configuration is injected at runtime, usually from the
`CLIMACOMMS_DEVICE` and `CLIMACOMMS_CONTEXT` environment variables.

## Devices and contexts as dispatch tags

Devices ([`ClimaComms.AbstractDevice`](@ref)) are
[singletons](https://docs.julialang.org/en/v1/manual/types/#man-singleton-types):
empty structs that carry no data. They exist so that Julia's multiple
dispatch can select the right implementation of a function. For example, a
package can provide CPU and GPU methods of the same operation:

```julia
import ClimaComms: AbstractCPUDevice, CUDADevice
import CUDA

my_allocate(::AbstractCPUDevice, data) = Array(data)
my_allocate(::CUDADevice, data) = CUDA.CuArray(data)
```

Low-level CliMA code (e.g., in
[ClimaCore](https://github.com/CliMA/ClimaCore.jl)) sometimes implements
device-specific methods like these. Higher-level code rarely needs to: it
interacts with devices through the device-flexible operations that
`ClimaComms` already provides, such as [`ClimaComms.array_type`](@ref),
[`ClimaComms.@time`](@ref), [`ClimaComms.@sync`](@ref), and
[`ClimaComms.@threaded`](@ref).

Contexts ([`ClimaComms.AbstractCommsContext`](@ref)) extend the same idea
to communication. A context wraps a device and, for distributed runs, the
communicator. Communication primitives — [`ClimaComms.reduce`](@ref),
[`ClimaComms.gather`](@ref), [`ClimaComms.bcast`](@ref),
[`ClimaComms.barrier`](@ref), and friends — dispatch on the context type:
the MPI methods forward to the corresponding MPI operations, while the
singleton methods are no-ops. Code written for the distributed case
therefore runs, without penalty or modification, in the single-process
case.

This "write for the parallel case, run anywhere" stance is the recommended
style throughout CliMA: obtain a context once, near the top of the program,
and pass it (or its device) down to the code that needs it.

## Backends as package extensions

`ClimaComms` itself depends only on `Adapt.jl` and Julia's logging
packages. The CUDA and MPI implementations live in [package
extensions](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions))
(`ext/ClimaCommsCUDAExt.jl` and `ext/ClimaCommsMPIExt.jl`) that Julia loads
automatically when `CUDA.jl` or `MPI.jl` is imported. There are two reasons
for this design:

- optional heavyweight packages such as `CUDA.jl` and `MPI.jl` should not
  be hard dependencies of the entire CliMA stack;
- keeping backend code out of the main package reduces load time for
  `ClimaComms` and every package downstream of it.

The cost of this design is that someone has to import the backend package.
Since driver scripts should not hard-code backends, `ClimaComms` provides
[`ClimaComms.@import_required_backends`](@ref), which imports `MPI.jl`
and/or `CUDA.jl` exactly when the environment variables request them. If a
required backend is missing, [`ClimaComms.device`](@ref) and
[`ClimaComms.context`](@ref) fail with an actionable error message.

!!! warning
    Only use [`ClimaComms.@import_required_backends`](@ref) in scripts,
    never in library code (i.e., in `src`): libraries should not assume
    that the backends are installed. If you implement device- or
    context-specific features in a package, put the backend-specific code
    in a package extension, as `ClimaComms` itself does.

## How CliMA packages use ClimaComms

`ClimaComms` sits at the very bottom of the CliMA software stack; nearly
every CliMA package accepts a device or context argument.

- [ClimaCore.jl](https://github.com/CliMA/ClimaCore.jl), the dynamical-core
  toolkit, takes a context when constructing distributed grids and spaces.
  The context's device determines where field data lives (CPU or GPU
  memory), and the context's graph exchange
  ([`ClimaComms.graph_context`](@ref)) fills the ghost (halo) regions of
  domain-decomposed fields during distributed simulations.
- [ClimaAtmos.jl](https://github.com/CliMA/ClimaAtmos.jl),
  [ClimaLand.jl](https://github.com/CliMA/ClimaLand.jl), and
  [ClimaOcean.jl](https://github.com/CliMA/ClimaOcean.jl) driver scripts
  select their hardware configuration through the `ClimaComms` environment
  variables, so the same experiment configuration moves from a laptop to a
  GPU cluster unchanged.
- [ClimaCoupler.jl](https://github.com/CliMA/ClimaCoupler.jl) passes one
  shared context to the component models it couples, keeping atmosphere,
  land, and ocean on a consistent set of processes and devices.
- Utility packages such as
  [ClimaTimeSteppers.jl](https://github.com/CliMA/ClimaTimeSteppers.jl) and
  [ClimaDiagnostics.jl](https://github.com/CliMA/ClimaDiagnostics.jl) use
  the device-flexible operations (e.g., [`ClimaComms.@threaded`](@ref),
  [`ClimaComms.allowscalar`](@ref)) to stay agnostic about where their
  inputs live, and use context primitives (e.g.,
  [`ClimaComms.iamroot`](@ref), [`ClimaComms.reduce`](@ref)) for I/O and
  reporting in distributed runs.

The pattern to imitate when developing with `ClimaComms`: accept a context
(or device) as an argument, dispatch on it where implementations must
differ, use the device-flexible primitives where they suffice, and leave
the choice of configuration to the top-level driver.
