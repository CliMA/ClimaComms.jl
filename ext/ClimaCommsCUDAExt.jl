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

end
