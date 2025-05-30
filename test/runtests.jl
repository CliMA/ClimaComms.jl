using Test
using ClimaComms
import ClimaComms: MArray

ClimaComms.@import_required_backends

context = ClimaComms.context()
pid, nprocs = ClimaComms.init(context)
device = ClimaComms.device(context)
AT = ClimaComms.array_type(device)

if ClimaComms.iamroot(context)
    @info "Running test" context device AT
end

if haskey(ENV, "CLIMACOMMS_TEST_DEVICE")
    if ENV["CLIMACOMMS_TEST_DEVICE"] == "CPU"
        @test device isa ClimaComms.AbstractCPUDevice
    elseif ENV["CLIMACOMMS_TEST_DEVICE"] == "CPUSingleThreaded"
        @test device isa ClimaComms.CPUSingleThreaded
    elseif ENV["CLIMACOMMS_TEST_DEVICE"] == "CPUMultiThreaded"
        @test device isa ClimaComms.CPUMultiThreaded
    elseif ENV["CLIMACOMMS_TEST_DEVICE"] == "CUDA"
        @test device isa ClimaComms.CUDADevice
    elseif ENV["CLIMACOMMS_TEST_DEVICE"] == "Metal"
        @test device isa ClimaComms.MetalDevice
    end
end

using SafeTestsets
@safetestset "macro hygiene" begin
    include("hygiene.jl")
end

if context isa ClimaComms.MPICommsContext
    graph_opt_list = [(; persistent = true), (; persistent = false)]
else
    graph_opt_list = [()]
end
@testset "tree test $graph_opt" for graph_opt in graph_opt_list
    for FT in (Float32, Float64)
        if device isa ClimaComms.MetalDevice && FT == Float64
            @info "Skipping Float64 test for Metal device"
            continue
        end

        # every process communicates with the root
        if ClimaComms.iamroot(context)
            # send 2*n items to the nth pid, receive 3*n
            sendpids = collect(2:nprocs)
            sendlengths = [2 * dest for dest in sendpids]
            sendarray = AT(fill(zero(FT), (2, sum(sendpids))))
            recvpids = collect(2:nprocs)
            recvlengths = [3 * dest for dest in recvpids]
            recvarray = AT(fill(zero(FT), (3, sum(recvpids))))
        else
            # send 3*pid items to the 1st pid, receive 2*pid
            sendpids = [1]
            sendlengths = [3 * pid]
            sendarray = AT(fill(zero(FT), (3, pid)))
            recvpids = [1]
            recvlengths = [2 * pid]
            recvarray = AT(fill(zero(FT), (2, pid)))
        end
        graph_context = ClimaComms.graph_context(
            context,
            sendarray,
            sendlengths,
            sendpids,
            recvarray,
            recvlengths,
            recvpids;
            graph_opt...,
        )

        # 1) fill buffers with pid
        fill!(sendarray, FT(pid))

        ClimaComms.start(graph_context)
        ClimaComms.progress(graph_context)
        ClimaComms.finish(graph_context)

        if ClimaComms.iamroot(context)
            offset = 0
            for s in 2:nprocs
                @test all(
                    ==(FT(s)),
                    view(recvarray, :, (offset + 1):(offset + s)),
                )
                offset += s
            end
        else
            @test all(==(FT(1)), recvarray)
        end

        # 2) send everything back
        if ClimaComms.iamroot(context)
            sendarray .= view(recvarray, 1:2, :)
        else
            sendarray .= FT(1)
        end

        ClimaComms.start(graph_context)
        ClimaComms.progress(graph_context)
        ClimaComms.finish(graph_context)

        @test all(==(FT(pid)), recvarray)
    end
end

@testset "linear test $graph_opt" for graph_opt in graph_opt_list
    for FT in (Float32, Float64)
        if device isa ClimaComms.MetalDevice && FT == Float64
            @info "Skipping Float64 test for Metal device"
            continue
        end
        # send 2 values up
        if pid < nprocs
            sendpids = Int[pid + 1]
            sendlengths = Int[2]
            sendarray = AT(fill(zero(FT), (2,)))
        else
            sendpids = Int[]
            sendlengths = Int[]
            sendarray = AT(fill(zero(FT), (0,)))
        end
        if pid > 1
            recvpids = Int[pid - 1]
            recvlengths = Int[2]
            recvarray = AT(fill(zero(FT), (2,)))
        else
            recvpids = Int[]
            recvlengths = Int[]
            recvarray = AT(fill(zero(FT), (0,)))
        end
        graph_context = ClimaComms.graph_context(
            context,
            sendarray,
            sendlengths,
            sendpids,
            recvarray,
            recvlengths,
            recvpids;
            graph_opt...,
        )

        # 1) send pid
        if pid < nprocs
            sendarray .= FT(pid)
        end
        ClimaComms.start(graph_context)
        ClimaComms.progress(graph_context)
        ClimaComms.finish(graph_context)

        if pid > 1
            @test all(==(FT(pid - 1)), recvarray)
        end

        # 2) send next
        if 1 < pid < nprocs
            sendarray .= recvarray
        end

        ClimaComms.start(graph_context)
        ClimaComms.progress(graph_context)
        ClimaComms.finish(graph_context)

        if pid > 2
            @test all(==(FT(pid - 2)), recvarray)
        end
    end
end

@testset "gather" begin
    for FT in (Float32, Float64)
        if device isa ClimaComms.MetalDevice && FT == Float64
            @info "Skipping Float64 test for Metal device"
            continue
        end
        local_array = AT(fill(FT(pid), (3, pid)))
        gathered = ClimaComms.gather(context, local_array)
        if ClimaComms.iamroot(context)
            @test gathered isa AT
            @test gathered ==
                  AT(reduce(hcat, [fill(FT(i), (3, i)) for i in 1:nprocs]))
        else
            @test isnothing(gathered)
        end
    end
end

@testset "reduce/reduce!/allreduce" begin
    for FT in (Float32, Float64)
        if device isa ClimaComms.MetalDevice && FT == Float64
            @info "Skipping Float64 test for Metal device"
            continue
        end
        pidsum = div(nprocs * (nprocs + 1), 2)

        sendrecvbuf = AT(fill(FT(pid), 3))
        ClimaComms.allreduce!(context, sendrecvbuf, +)
        @test sendrecvbuf == AT(fill(FT(pidsum), 3))

        sendrecvbuf = AT(fill(FT(pid), 3))
        ClimaComms.reduce!(context, sendrecvbuf, +)
        if ClimaComms.iamroot(context)
            @test sendrecvbuf == AT(fill(FT(pidsum), 3))
        end

        sendbuf = AT(fill(FT(pid), 2))
        recvbuf = AT(zeros(FT, 2))
        ClimaComms.reduce!(context, sendbuf, recvbuf, +)
        if ClimaComms.iamroot(context)
            @test recvbuf == AT(fill(FT(pidsum), 2))
        end

        sendbuf = AT(fill(FT(pid), 2))
        recvbuf = AT(zeros(FT, 2))
        ClimaComms.allreduce!(context, sendbuf, recvbuf, +)
        @test recvbuf == AT(fill(FT(pidsum), 2))

        recvval = ClimaComms.allreduce(context, FT(pid), +)
        @test recvval == FT(pidsum)

        recvval = ClimaComms.reduce(context, FT(pid), +)
        if ClimaComms.iamroot(context)
            @test recvval == FT(pidsum)
        end
    end
end

@testset "bcast" begin
    @test ClimaComms.bcast(context, ClimaComms.iamroot(context))
    @test ClimaComms.bcast(context, pid) == 1
    @test ClimaComms.bcast(context, "root pid is $pid") == "root pid is 1"
    @test ClimaComms.bcast(context, AT(fill(Float32(pid), 3))) ==
          AT(fill(Float32(1), 3))

    if !(device isa ClimaComms.MetalDevice)
        @test ClimaComms.bcast(context, AT(fill(Float64(pid), 3))) ==
              AT(fill(Float64(1), 3))
    end
end

@testset "allowscalar" begin
    a = AT(rand(Float32, 3))
    local x
    ClimaComms.allowscalar(device) do
        x = a[1]
    end
    (device isa ClimaComms.CUDADevice || device isa ClimaComms.MetalDevice) &&
        @test_throws ErrorException a[1]
    @test x == Array(a)[1]
end

@testset "independent threaded" begin
    FT = Float32
    a = AT(rand(FT, 100))
    b = AT(rand(FT, 100))
    is_single_cpu_thread =
        device isa ClimaComms.CPUSingleThreaded &&
        context isa ClimaComms.SingletonCommsContext

    kernel1!(a, b) = ClimaComms.@threaded for i in axes(a, 1)
        a[i] = b[i]
    end
    kernel1!(a, b)
    @test a == b
    is_single_cpu_thread && @test (@allocated kernel1!(a, b)) == 0

    kernel2!(a, b) = ClimaComms.@threaded coarsen=:static for i in axes(a, 1)
        a[i] = 2 * b[i]
    end
    kernel2!(a, b)
    @test a == 2 * b
    is_single_cpu_thread && @test (@allocated kernel2!(a, b)) == 0

    kernel3!(a, b) = ClimaComms.@threaded device coarsen=3 for i in axes(a, 1)
        a[i] = 3 * b[i]
    end
    kernel3!(a, b)
    @test a == 3 * b
    is_single_cpu_thread && @test (@allocated kernel3!(a, b)) == 0

    kernel4!(a, b) = ClimaComms.@threaded device coarsen=400 for i in axes(a, 1)
        a[i] = 4 * b[i]
    end
    kernel4!(a, b)
    @test a == 4 * b
    is_single_cpu_thread && @test (@allocated kernel4!(a, b)) == 0

    kernel5!(a, b) = ClimaComms.@threaded block_size=50 for i in axes(a, 1)
        a[i] = 5 * b[i]
    end
    kernel5!(a, b)
    @test a == 5 * b
    is_single_cpu_thread && @test (@allocated kernel5!(a, b)) == 0
end

@testset "interdependent threaded" begin
    set_deriv_at_point!(output, input, i) =
        if i == 1
            output[1] = input[2] - input[1]
        elseif i == 100
            output[100] = input[100] - input[99]
        else
            output[i] = (input[i + 1] - 2 * input[i] + input[i - 1]) / 2
        end

    function threaded_deriv_with_respect_to_i(device, input, i)
        output =
            ClimaComms.static_shared_memory_array(device, eltype(input), 100)
        ClimaComms.@sync_interdependent i set_deriv_at_point!(output, input, i)
        return output
    end

    function unthreaded_deriv_with_respect_to_i(device, input)
        output = MArray{Tuple{100}, eltype(input)}(undef)
        for i in axes(input, 1)
            set_deriv_at_point!(output, input, i)
        end
        return output
    end

    threaded_deriv3_with_respect_to_i!(device, a, b) =
        ClimaComms.@threaded device for i in @interdependent(axes(a, 1))
            ∂b_∂i = threaded_deriv_with_respect_to_i(device, b, i)
            ∂²b_∂i² = threaded_deriv_with_respect_to_i(device, ∂b_∂i, i)
            ∂³b_∂i³ = threaded_deriv_with_respect_to_i(device, ∂²b_∂i², i)
            ClimaComms.@sync_interdependent i a[i] = ∂³b_∂i³[i]
        end

    unthreaded_deriv3_with_respect_to_i!(device, a, b) =
        ClimaComms.allowscalar(device) do
            ∂b_∂i = unthreaded_deriv_with_respect_to_i(device, b)
            ∂²b_∂i² = unthreaded_deriv_with_respect_to_i(device, ∂b_∂i)
            ∂³b_∂i³ = unthreaded_deriv_with_respect_to_i(device, ∂²b_∂i²)
            for i in axes(a, 1)
                a[i] = ∂³b_∂i³[i]
            end
        end

    FT = Float32
    a_threaded = AT(rand(FT, 100))
    a_unthreaded = AT(rand(FT, 100))
    b = AT(rand(FT, 100))
    is_single_cpu_thread =
        device isa ClimaComms.CPUSingleThreaded &&
        context isa ClimaComms.SingletonCommsContext

    threaded_deriv3_with_respect_to_i!(device, a_threaded, b)
    unthreaded_deriv3_with_respect_to_i!(device, a_unthreaded, b)
    @test a_threaded == a_unthreaded

    # TODO: Figure out source of allocations for interdependent iterators.
    threaded_allocations =
        @allocated threaded_deriv3_with_respect_to_i!(device, a_threaded, b)
    @info "Allocated $threaded_allocations bytes"
    is_single_cpu_thread && @test_broken threaded_allocations == 0
end

@testset "independent and interdependent threaded" begin
    set_deriv_at_point!(output, input, i, j...) =
        if i == 1
            output[1] = input[2, j...] - input[1, j...]
        elseif i == 100
            output[100] = input[100, j...] - input[99, j...]
        else
            output[i] =
                (input[i + 1, j...] - 2 * input[i, j...] + input[i - 1, j...]) /
                2
        end

    function threaded_deriv_with_respect_to_i(device, input, i, j...)
        output =
            ClimaComms.static_shared_memory_array(device, eltype(input), 100)
        ClimaComms.@sync_interdependent i begin
            set_deriv_at_point!(output, input, i, j...)
        end
        return output
    end

    function unthreaded_deriv_with_respect_to_i(device, input, j...)
        output = MArray{Tuple{100}, eltype(input)}(undef)
        for i in axes(input, 1)
            set_deriv_at_point!(output, input, i, j...)
        end
        return output
    end

    threaded_deriv3_with_respect_to_i!(device, a, b) =
        ClimaComms.@threaded device begin
            for i in @interdependent(axes(a, 1)), j in axes(a, 2)
                ∂b_∂i = threaded_deriv_with_respect_to_i(device, b, i, j)
                ∂²b_∂i² = threaded_deriv_with_respect_to_i(device, ∂b_∂i, i)
                ∂³b_∂i³ = threaded_deriv_with_respect_to_i(device, ∂²b_∂i², i)
                ClimaComms.@sync_interdependent a[i, j] = ∂³b_∂i³[i]
            end
        end

    unthreaded_deriv3_with_respect_to_i!(device, a, b) =
        ClimaComms.allowscalar(device) do
            for j in axes(a, 2)
                ∂b_∂i = unthreaded_deriv_with_respect_to_i(device, b, j)
                ∂²b_∂i² = unthreaded_deriv_with_respect_to_i(device, ∂b_∂i)
                ∂³b_∂i³ = unthreaded_deriv_with_respect_to_i(device, ∂²b_∂i²)
                for i in axes(a, 1)
                    a[i, j] = ∂³b_∂i³[i]
                end
            end
        end

    FT = Float32
    a_threaded = AT(rand(FT, 100, 100))
    a_unthreaded = AT(rand(FT, 100, 100))
    b = AT(rand(FT, 100, 100))
    is_single_cpu_thread =
        device isa ClimaComms.CPUSingleThreaded &&
        context isa ClimaComms.SingletonCommsContext

    threaded_deriv3_with_respect_to_i!(device, a_threaded, b)
    unthreaded_deriv3_with_respect_to_i!(device, a_unthreaded, b)
    @test a_threaded == a_unthreaded

    # TODO: Figure out source of allocations for interdependent iterators.
    threaded_allocations =
        @allocated threaded_deriv3_with_respect_to_i!(device, a_threaded, b)
    @info "Allocated $threaded_allocations bytes"
    is_single_cpu_thread && @test_broken threaded_allocations == 0
end

import Adapt
@testset "Adapt" begin
    @test Adapt.adapt(Array, ClimaComms.CUDADevice()) ==
          ClimaComms.CPUSingleThreaded()
    @static if ClimaComms.device() isa ClimaComms.CUDADevice
        @test Adapt.adapt(Array, ClimaComms.CUDADevice()) ==
              ClimaComms.CPUSingleThreaded()
        @test Adapt.adapt(CUDA.CuArray, ClimaComms.CUDADevice()) ==
              ClimaComms.CUDADevice()
        @test Adapt.adapt(CUDA.CuArray, ClimaComms.CPUSingleThreaded()) ==
              ClimaComms.CUDADevice()
    end

    @static if ClimaComms.device() isa ClimaComms.MetalDevice
        @test Adapt.adapt(Array, ClimaComms.MetalDevice()) ==
              ClimaComms.CPUSingleThreaded()
        @test Adapt.adapt(Metal.MtlArray, ClimaComms.MetalDevice()) ==
              ClimaComms.MetalDevice()
        @test Adapt.adapt(Metal.MtlArray, ClimaComms.CPUSingleThreaded()) ==
              ClimaComms.MetalDevice()
    end

    @test Adapt.adapt(Array, ClimaComms.context(ClimaComms.CUDADevice())) ==
          ClimaComms.context(ClimaComms.CPUSingleThreaded())
    @static if ClimaComms.device() isa ClimaComms.CUDADevice
        @test Adapt.adapt(Array, ClimaComms.context(ClimaComms.CUDADevice())) ==
              ClimaComms.context(ClimaComms.CPUSingleThreaded())
        @test Adapt.adapt(
            CUDA.CuArray,
            ClimaComms.context(ClimaComms.CUDADevice()),
        ) == ClimaComms.context(ClimaComms.CUDADevice())
        @test Adapt.adapt(
            CUDA.CuArray,
            ClimaComms.context(ClimaComms.CPUSingleThreaded()),
        ) == ClimaComms.context(ClimaComms.CUDADevice())
    end

    @static if ClimaComms.device() isa ClimaComms.MetalDevice
        @test Adapt.adapt(
            Array,
            ClimaComms.context(ClimaComms.MetalDevice()),
        ) == ClimaComms.context(ClimaComms.CPUSingleThreaded())
        @test Adapt.adapt(
            Metal.MtlArray,
            ClimaComms.context(ClimaComms.CPUSingleThreaded()),
        ) == ClimaComms.context(ClimaComms.MetalDevice())
        @test Adapt.adapt(
            Metal.MtlArray,
            ClimaComms.context(ClimaComms.MetalDevice()),
        ) == ClimaComms.context(ClimaComms.MetalDevice())
    end
end

@testset "Logging" begin
    include("logging.jl")
end
