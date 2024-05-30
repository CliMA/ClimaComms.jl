ClimaComms.jl Release Notes
========================

v0.7.0
-------

- ![][badge-💥breaking] `ClimaComms.@import_required_backends` was removed, as there were code loading issues. It is now recommended to use the following code loading pattern:
  ```julia
  ClimaComms.cuda_is_required() && import CUDA
  ClimaComms.mpi_is_required() && import MPI
  ```

v0.6.0
-------

- ![][badge-💥breaking] `ClimaComms` does no longer try to guess the correct
  compute device: the default is now CPU. To control which device to use,
  use the `CLIMACOMMS_DEVICE` environment variable.
- ![][badge-💥breaking] `CUDA` and `MPI` are now extensions in `ClimaComms`. To
  use `CUDA`/`MPI`, `CUDA.jl`/`MPI.jl` have to be loaded. A convenience macro
  `ClimaComms.@import_required_backends` checks what device/context could be
  used and conditionally loads `CUDA.jl`/`MPI.jl`. It is recommended to change
  ```julia
  import ClimaComms
  ```
  to 
  ```julia
  import ClimaComms
  ClimaComms.@import_required_backends
  ```
  This has to be done before calling `ClimaComms.context()`.

[badge-💥breaking]: https://img.shields.io/badge/💥BREAKING-red.svg
