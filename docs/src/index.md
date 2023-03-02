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
ClimaComms.CPUDevice
ClimaComms.CUDADevice
ClimaComms.device
ClimaComms.array_type
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

## Communication interface

```@docs
ClimaComms.init
ClimaComms.mypid
ClimaComms.iamroot
ClimaComms.nprocs
ClimaComms.barrier
ClimaComms.reduce
ClimaComms.allreduce
ClimaComms.allreduce!
ClimaComms.abort
ClimaComms.start
ClimaComms.progress
ClimaComms.finish
```
