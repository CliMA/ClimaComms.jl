import ClimaComms, MPI
using Logging, Test

ctx = ClimaComms.context()
(mypid, nprocs) = ClimaComms.init(ctx)

@testset "MPIFileLogger" begin
    log_dir = mktempdir()
    logger = ClimaComms.MPIFileLogger(ctx, log_dir)
    with_logger(logger) do
        test_str = "Test message from rank $mypid"
        @info test_str
        log_content = read(joinpath(log_dir, "rank_$mypid.log"), String)
        @test occursin(test_str, log_content)
    end
end

@testset "MPILogger" begin
    io = IOBuffer()
    logger = ClimaComms.MPILogger(io, ctx)
    with_logger(logger) do
        @info "smoke test"
    end
    str = String(take!(io))
    @test contains(str, "[P$mypid]  Info: smoke test\n")

    # Test with file IOStream
    test_filename, io = mktemp()
    logger = ClimaComms.MPILogger(io, ctx)
    with_logger(logger) do
        test_str = "Test message from rank $mypid"
        @info test_str
        flush(io)
        close(io)

        log_content = read(test_filename, String)
        @test occursin(test_str, log_content)
        @test occursin(test_str, log_content)
    end
end
