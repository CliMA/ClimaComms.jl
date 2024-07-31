# ClimaComms

```@meta
CurrentModule = ClimaComms
```

```@docs
ClimaComms
```

## Loading

```@docs
ClimaComms.import_required_backends
ClimaComms.cuda_is_required
ClimaComms.mpi_is_required
```

## Devices

```@docs
ClimaComms.AbstractDevice
ClimaComms.AbstractCPUDevice
ClimaComms.CPUSingleThreading
ClimaComms.CPUMultiThreading
ClimaComms.CUDADevice
ClimaComms.device
ClimaComms.device_functional
ClimaComms.array_type
ClimaComms.allowscalar
ClimaComms.@threaded
ClimaComms.@time
ClimaComms.@elapsed
ClimaComms.@assert
ClimaComms.@sync
ClimaComms.@cuda_sync
```

## Contexts

```@docs
ClimaComms.AbstractCommsContext
ClimaComms.SingletonCommsContext
ClimaComms.MPICommsContext
ClimaComms.AbstractGraphContext
ClimaComms.context
ClimaComms.graph_context
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
