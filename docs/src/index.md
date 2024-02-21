# ClimaComms

```@meta
CurrentModule = ClimaComms
```

```@docs
ClimaComms
```

## Devices

```@docs
ClimaComms.AbstractDevice
ClimaComms.AbstractCPUDevice
ClimaComms.CPUSingleThreading
ClimaComms.CPUMultiThreading
ClimaComms.CUDADevice
ClimaComms.device
ClimaComms.array_type
ClimaComms.@threaded
ClimaComms.@time
ClimaComms.@elapsed
ClimaComms.@sync
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
