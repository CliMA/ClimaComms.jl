# APIs

```@meta
CurrentModule = ClimaComms
```

```@docs
ClimaComms
```

## Loading

```@docs
ClimaComms.@import_required_backends
ClimaComms.cuda_is_required
ClimaComms.mpi_is_required
```

## Devices

```@docs
ClimaComms.AbstractDevice
ClimaComms.AbstractCPUDevice
ClimaComms.CPUSingleThreaded
ClimaComms.CPUMultiThreaded
ClimaComms.CUDADevice
ClimaComms.device
ClimaComms.device_functional
ClimaComms.array_type
ClimaComms.free_memory
ClimaComms.total_memory
ClimaComms.allowscalar
ClimaComms.@time
ClimaComms.@elapsed
ClimaComms.@assert
ClimaComms.@sync
ClimaComms.@cuda_sync
Adapt.adapt_structure(::Type{<:AbstractArray}, ::ClimaComms.AbstractDevice)
```

## Threading

```@docs
ClimaComms.@threaded
ClimaComms.@shmem_threaded
ClimaComms.threaded
ClimaComms.threadable
ClimaComms.shareable
ClimaComms.set_metadata
ClimaComms.disable_auto_sync
ClimaComms.auto_sync!
ClimaComms.sync_shmem_threads!
ClimaComms.shmem_array
ClimaComms.@unique_shmem_thread
ClimaComms.unique_shmem_thread
ClimaComms.shmem_reduce!
ClimaComms.shmem_mapreduce!
ClimaComms.shmem_any!
ClimaComms.shmem_all!
ClimaComms.shmem_sum!
ClimaComms.shmem_prod!
ClimaComms.shmem_maximum!
ClimaComms.shmem_minimum!
```

## Contexts

```@docs
ClimaComms.AbstractCommsContext
ClimaComms.SingletonCommsContext
ClimaComms.MPICommsContext
ClimaComms.AbstractGraphContext
ClimaComms.context
ClimaComms.graph_context
Adapt.adapt_structure(::Type{<:AbstractArray}, ::ClimaComms.AbstractCommsContext)
```

## Logging

```@docs
ClimaComms.OnlyRootLogger
ClimaComms.MPILogger
ClimaComms.FileLogger
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
```

### Graph exchange

```@docs
ClimaComms.start
ClimaComms.progress
ClimaComms.finish
```

