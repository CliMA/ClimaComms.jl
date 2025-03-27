using Test
using ClimaComms
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
    @test ClimaComms.bcast(context, AT(fill(Float64(pid), 3))) ==
          AT(fill(Float64(1), 3))
end

@testset "allowscalar" begin
    a = AT(rand(3))
    local x
    ClimaComms.allowscalar(device) do
        x = a[1]
    end
    device isa ClimaComms.CUDADevice && @test_throws ErrorException a[1]
    @test x == Array(a)[1]
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
end

@testset "Logging" begin
    include("logging.jl")
end

@testset "Adapt" begin
    include("adapt.jl")
end
