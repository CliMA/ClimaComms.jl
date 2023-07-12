module BenchmarkToolsExt

isdefined(Base, :get_extension) ? (import BenchmarkTools) :
(import ..BenchmarkTools)

import CUDA

#=
This file is conditionally loaded
if BenchmarkTools has been loaded:

```julia
using BenchmarkTools # to load this file
using ClimaComms
```
=#

"""
    @benchmark device expr

Device-flexible `@benchmark`.

Lowers to
```julia
BenchmarkTools.@benchmark expr
```
for CPU devices and
```julia
BenchmarkTools.@benchmark CUDA.@async expr
```
for CUDA devices.
"""
macro benchmark(device, expr)
    return quote
        if $(esc(device)) isa CUDADevice
            BenchmarkTools.@benchmark CUDA.@async $(expr)
        else
            @assert $(esc(device)) isa AbstractDevice
            BenchmarkTools.@benchmark $(expr)
        end
    end
end

end
