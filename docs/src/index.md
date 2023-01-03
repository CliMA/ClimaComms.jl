# ClimaComms

```@meta
CurrentModule = ClimaComms
```

```@docs
ClimaComms
```

### Contexts
```@docs
ClimaComms.AbstractCommsContext
ClimaComms.AbstractGraphContext
```

### Devices
```@docs
ClimaComms.AbstractDevice
ClimaComms.CPU
ClimaComms.CUDA
```

### Communication interface
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
ClimaComms.graph_context
ClimaComms.start
ClimaComms.progress
ClimaComms.finish
```

### Contexts

```@docs
ClimaComms.SingletonCommsContext
```
