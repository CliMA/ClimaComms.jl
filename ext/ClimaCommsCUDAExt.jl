module ClimaCommsCUDAExt

import CUDA

import ClimaComms

function ClimaComms._assign_device(::ClimaComms.CUDADevice, rank_number)
    CUDA.device!(rank_number % CUDA.ndevices())
    return nothing
end

function ClimaComms.device_functional(::ClimaComms.CUDADevice)
    return CUDA.functional()
end

ClimaComms.array_type(::ClimaComms.CUDADevice) = CUDA.CuArray
ClimaComms.allowscalar(f, ::ClimaComms.CUDADevice, args...; kwargs...) =
    CUDA.@allowscalar f(args...; kwargs...)
ClimaComms.fill(::CUDADevice, value, dims...) = CUDA.fill(value, dims...)

# Extending ClimaComms methods that operate on expressions (cannot use dispatch here)
ClimaComms.cuda_sync(expr) = CUDA.@sync expr
ClimaComms.cuda_time(expr) = CUDA.@time expr
ClimaComms.cuda_elasped(expr) = CUDA.@elapsed expr

end
