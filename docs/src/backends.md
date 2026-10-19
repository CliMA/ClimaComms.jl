# Implementing a New Backend

A device in `ClimaComms` is a singleton struct, and each operation that differs between devices is a method of a `ClimaComms` function that dispatches on the device type.
A backend package, such as `CUDA.jl`, is a weak dependency, and the methods that call it are in a [package extension](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions)) that Julia loads when the backend package is imported.

The function `test_device_interface(device)` in `test/device_interface.jl` tests the methods of the extension that this page lists.
The test suite calls it for the device that `CLIMACOMMS_DEVICE` selects.

## Definitions in `src`

A device that `CLIMACOMMS_DEVICE` can select requires these definitions:

- the device type in `src/devices.jl`, a subtype of `AbstractGPUDevice` for a GPU;
- a value of `CLIMACOMMS_DEVICE` in `device_type`, and a check in `device` that the extension is loaded;
- the functions `<backend>_is_required` and `<backend>_ext_is_loaded` in `src/loading.jl`, and a block of `@import_required_backends` that imports the backend package;
- a branch of `_backend_hint` in `src/ClimaComms.jl`, which names the backend package in the message of a `MethodError`;
- the backend package in `[weakdeps]` and `[compat]` of `Project.toml`, and the extension in `[extensions]`;
- a value of `CLIMACOMMS_TEST_DEVICE` in `test/runtests.jl`, and a CI job that runs the tests on the device, if CliMA has a runner for it.

A device that does not support `Float64` also defines a method of [`supports_float64`](@ref ClimaComms.supports_float64) that returns `false`.

## Methods in the extension

The extension, `ext/ClimaComms<Backend>Ext.jl`, defines these methods for the device type:

| method                                        | returns or does                                                         |
| --------------------------------------------- | ----------------------------------------------------------------------- |
| `array_type(device)`                          | the array type of the device                                            |
| `device_functional(device)`                   | `true` if the device can run code                                       |
| `free_memory(device)`, `total_memory(device)` | the bytes of memory that are available for allocation now and in total  |
| `device_synchronize(device)`                  | waits for all work queued on the device                                 |
| `allowscalar(f, device, args...; kwargs...)`  | `f(args...; kwargs...)`, with scalar indexing of device arrays allowed  |
| `time`, `elapsed`, `sync`, `device_sync`      | the functional forms of `@time`, `@elapsed`, `@sync`, and `@device_sync` |
| `assert(device, cond, text)`                  | the functional form of `@assert`                                        |
| `Adapt.adapt_structure(::Type{<:A}, ::AbstractDevice)`, where `A` is the array type     |         the device            |

An extension of a device that is used with MPI also defines `_assign_device(device, id)`, which selects the device of each rank.

The extension can also define these methods, which have generic definitions in `src`:

| method                                                        | returns or does                                                         |
| ------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `Base.summary(io, device)`                                    | prints a description of the device to `io`                              |
| `Adapt.adapt_structure(::Type{<:A}, ::AbstractCommsContext)`  | a context on the device, of the type that `CLIMACOMMS_CONTEXT` selects  |

## Threaded loops on a GPU

ClimaComms defines the methods of [`@threaded`](@ref ClimaComms.@threaded) for every `AbstractGPUDevice`.
They are built on six functions, which the extension defines for the device type:

- [`compile_kernel`](@ref ClimaComms.compile_kernel)`(device, f)`;
- [`launch_kernel`](@ref ClimaComms.launch_kernel)`(device, kernel, blocks, threads_in_block)`;
- [`max_blocks_in_kernel`](@ref ClimaComms.max_blocks_in_kernel)`(device, kernel)`;
- [`max_threads_in_block`](@ref ClimaComms.max_threads_in_block)`(device, kernel, block_size, n_threads)`;
- [`thread_index_in_kernel`](@ref ClimaComms.thread_index_in_kernel)`(device)` and
  [`threads_in_kernel`](@ref ClimaComms.threads_in_kernel)`(device)`, which are called from device code.

`compile_kernel` can return any object that the other functions accept.
The Metal extension keeps `f` with the compiled kernel and passes it as the argument of each launch, because Metal.jl makes the arrays that `f` captures accessible to the GPU only when `f` is a launch argument.

The CUDA extension, `ext/ClimaCommsCUDAExt.jl`, defines the methods of the section "Methods in the extension" and the six functions for `CUDADevice`, and the Metal extension, `ext/ClimaCommsMetalExt.jl`, for `MetalDevice`.
