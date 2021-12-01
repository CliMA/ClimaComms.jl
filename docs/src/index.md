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
```

### Communication interface
```@docs
ClimaComms.init
ClimaComms.mypid
ClimaComms.iamroot
ClimaComms.nprocs
ClimaComms.singlebuffered
ClimaComms.start
ClimaComms.progress
ClimaComms.finish
ClimaComms.barrier
ClimaComms.reduce
ClimaComms.abort
```

### Neighbors
```@docs
ClimaComms.Neighbor
```

```@docs
ClimaComms.id
ClimaComms.send_stage
ClimaComms.recv_stage
```
