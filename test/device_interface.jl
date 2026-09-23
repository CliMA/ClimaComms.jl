using Test
import Adapt
import ClimaComms

"""
    test_device_interface(device)

Test the methods that ClimaComms requires of every `device`, and the functions
that the extension of a GPU backend defines for an `AbstractGPUDevice`.
"""
function test_device_interface(device)
    AT = ClimaComms.array_type(device)
    FT = ClimaComms.supports_float64(device) ? Float64 : Float32
    @test ClimaComms.device_functional(device)
    @test 0 <= ClimaComms.free_memory(device) <= ClimaComms.total_memory(device)
    @test !isempty(summary(device))
    @test ClimaComms.supports_float64(device) isa Bool
    @test Adapt.adapt(Array, device) == ClimaComms.CPUSingleThreaded()
    @test isnothing(ClimaComms.device_synchronize(device))

    a = AT(zeros(FT, 3))
    @test ClimaComms.allowscalar(getindex, device, a, 1) == 0
    @test ClimaComms.cuda_sync(fill!, device, a, 1) == AT(ones(FT, 3))
    @test ClimaComms.sync(fill!, device, a, 2) == AT(fill(FT(2), 3))
    @test ClimaComms.elapsed(fill!, device, a, 3) >= 0
    @test ClimaComms.time(fill!, device, a, 4) == AT(fill(FT(4), 3))

    double!(x, flag) = ClimaComms.@threaded device for i in axes(x, 1)
        ClimaComms.@assert device flag "flag must be true at index $i"
        x[i] = 2 * x[i]
    end
    double!(a, true)
    @test Array(a) == fill(8, 3)
    # A failed assertion in a kernel makes a CUDA context unusable until the
    # process exits.
    device isa ClimaComms.CUDADevice ||
        @test_throws Exception ClimaComms.@cuda_sync device double!(a, false)

    device isa ClimaComms.AbstractGPUDevice || return nothing
    @test Adapt.adapt(AT, ClimaComms.CPUSingleThreaded()) == device
    @test Adapt.adapt(AT, ClimaComms.context(ClimaComms.CPUSingleThreaded())) ==
          ClimaComms.context(device)
    @test which(ClimaComms.device_synchronize, (typeof(device),)) !=
          which(ClimaComms.device_synchronize, (ClimaComms.AbstractDevice,))
    b = AT{FT}(undef, 2^16)
    function write_threads_in_kernel()
        i = ClimaComms.thread_index_in_kernel(device)
        b[i] = ClimaComms.threads_in_kernel(device)
        return nothing
    end
    kernel = ClimaComms.compile_kernel(device, write_threads_in_kernel)
    @test ClimaComms.max_blocks_in_kernel(device, kernel) >= 2^8
    @test ClimaComms.max_threads_in_block(device, kernel, Val(:auto), 2^24) >=
          2^8
    @test ClimaComms.max_threads_in_block(device, kernel, 2^8, 2^16) == 2^8
    ClimaComms.launch_kernel(device, kernel, 2^8, 2^8)
    ClimaComms.device_synchronize(device)
    @test Array(b) == fill(2^16, 2^16)
    return nothing
end
