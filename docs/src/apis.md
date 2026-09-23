# APIs

```@meta
CurrentModule = ClimaComms
```

```@docs
ClimaComms
```

## Loading backends

```@docs
ClimaComms.@import_required_backends
ClimaComms.cuda_is_required
ClimaComms.mpi_is_required
ClimaComms.cuda_ext_is_loaded
ClimaComms.mpi_ext_is_loaded
```

## Devices

```@docs
ClimaComms.AbstractDevice
ClimaComms.AbstractCPUDevice
ClimaComms.AbstractGPUDevice
ClimaComms.CPUSingleThreaded
ClimaComms.CPUMultiThreaded
ClimaComms.CUDADevice
ClimaComms.device
ClimaComms.device_functional
ClimaComms.array_type
ClimaComms.free_memory
ClimaComms.total_memory
ClimaComms.supports_float64
ClimaComms.device_synchronize
Adapt.adapt_structure(::Type{<:AbstractArray}, ::ClimaComms.AbstractDevice)
```

### Device-flexible operations

```@docs
ClimaComms.@time
ClimaComms.@elapsed
ClimaComms.@assert
ClimaComms.@sync
ClimaComms.@cuda_sync
ClimaComms.time
ClimaComms.elapsed
ClimaComms.sync
ClimaComms.cuda_sync
ClimaComms.allowscalar
```

### Threaded loops

```@docs
ClimaComms.@threaded
ClimaComms.threaded
ClimaComms.threadable
ClimaComms.ThreadableWrapper
```

### GPU backends

```@docs
ClimaComms.compile_kernel
ClimaComms.launch_kernel
ClimaComms.max_blocks_in_kernel
ClimaComms.max_threads_in_block
ClimaComms.thread_index_in_kernel
ClimaComms.threads_in_kernel
```

## Contexts

```@docs
ClimaComms.AbstractCommsContext
ClimaComms.SingletonCommsContext
ClimaComms.MPICommsContext
ClimaComms.context
ClimaComms.local_communicator
Adapt.adapt_structure(::Type{<:AbstractArray}, ::ClimaComms.AbstractCommsContext)
```

## Context operations

```@docs
ClimaComms.init
ClimaComms.mypid
ClimaComms.iamroot
ClimaComms.nprocs
ClimaComms.abort
```

## Collective operations

```@docs
ClimaComms.barrier
ClimaComms.reduce
ClimaComms.reduce!
ClimaComms.allreduce
ClimaComms.allreduce!
ClimaComms.bcast
ClimaComms.gather
```

### Graph exchange

```@docs
ClimaComms.AbstractGraphContext
ClimaComms.graph_context
ClimaComms.start
ClimaComms.progress
ClimaComms.finish
```

## Loggers

```@docs
ClimaComms.OnlyRootLogger
ClimaComms.MPILogger
ClimaComms.FileLogger
```

## Utilities

```@docs
ClimaComms.with_tempdir
```
