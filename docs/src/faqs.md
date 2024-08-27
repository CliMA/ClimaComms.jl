# Frequently Asked Questions

## How do I run my simulation on a GPU?

Set the environment variable `CLIMACOMMS_DEVICE` to `CUDA`. This can be
accomplished in your Julia script with (at the top)
```julia
ENV["CLIMACOMMS_DEVICE"] = "CUDA"
```
or calling
```julia
export CLIMACOMMS_DEVICE="CUDA"
```
in your shell (outside of Julia, no spaces).

## My simulation does not start and crashes with a `MPI` error. I don't want to run with `MPI`. What should I do?

`ClimaComms` tries to be smart and select the best configuration for your run.
Sometimes, it fails. In this case, you can force `ClimaComms` to ignore `MPI`
with
```julia
ENV["CLIMACOMMS_CONTEXT"] = "SINGLETON"
```
at the top of your Julia script or by calling
```julia
export CLIMACOMMS_CONTEXT="SINGLETON"
```
in your shell (outside of Julia, no spaces).

## My code is saying something about `ClimaComms.@import_required_backends`, what does that mean?

When you are using the environment variables to control the execution of your
script, `ClimaComms` can detect that some important packages are not loaded. For
example, `ClimaComms` will emit an error if you set `CLIMACOMMS_DEVICE="CUDA"`
but do not import `CUDA.jl` in your code.

`ClimaComms` provides a macro, [`ClimaComms.@import_required_backends`](@ref),
that you can add at the top of your scripts to automatically load the required
packages when needed. Note, the packages have to be in your Julia environment,
so you might install packages like ` MPI.jl` and `CUDA.jl`.

