# Stencil: applies a space-invariant, linear, symmetric filter (stencil)
# to a square grid.
#
# Include this file after setting `Context` to one of the `ClimaComms.jl`
# backends' context types.

using CUDA
using Logging
using OffsetArrays
using Printf
using Test
using ClimaComms

# defaults for grid dimension, stencil radius, etc.
const default_n = 256
const default_radius = 2
const default_niterations = 50

# floating point precisions to test with
const default_float_types = (Float64, Float32)

# array types with which to test
#const default_array_types = CUDA.has_cuda() ? (CuArray, Array) : (Array,)
const default_array_types = (Array,)

puts(s...) = ccall(:puts, Cint, (Cstring,), string(s...))

epsilon(::Type{Float64}) = 1.0e-8
epsilon(::Type{Float32}) = 0.0001f0
coefx(::Type{Float64}) = 1.0
coefx(::Type{Float32}) = 1.0f0
coefy(::Type{Float64}) = 1.0
coefy(::Type{Float32}) = 1.0f0

# find 2 factors of `n` closest to `√n`
function factor(n::Int)
    f1 = floor(Int, sqrt(n + 1.0))
    f2 = 0
    while f1 > 0
        if n % f1 == 0
            f2 = convert(Int, n / f1)
            break
        end
        f1 -= 1
    end
    return f1, f2
end

# partition `n` into `nr` parts and return the size and the offsets
# of pid `rdim`s part
function compute_dim(n::Int, nr::Int, rdim::Int)
    dim, dimleft = divrem(n, nr)
    if rdim < dimleft
        ostart = (dim + 1) * rdim + 1
        oend = ostart + dim
    else
        ostart = (dim + 1) * dimleft + dim * (rdim - dimleft) + 1
        oend = ostart + dim - 1
    end
    dim = oend - ostart + 1
    return dim, ostart, oend
end

# set up stencil weights to reflect a discrete divergence operator
function setup_stencil(AT, FT, radius)
    weight = zeros(FT, 2radius + 1, 2radius + 1)
    oweight = OffsetArray(weight, (-radius):radius, (-radius):radius)
    for i in 1:radius
        oweight[0, i] = oweight[i, 0] = 1.0 / (2.0 * i * radius)
        oweight[0, -i] = oweight[-i, 0] = -1.0 / (2.0 * i * radius)
    end
    return OffsetArray(AT(weight), (-radius):radius, (-radius):radius)
end

# set up input array with ghost elements
function setup_input(AT, FT, height, cstart, cend, width, rstart, rend, radius)
    input = zeros(FT, height + 2radius, width + 2radius)
    oinput = OffsetArray(
        input,
        (rstart - radius):(rend + radius),
        (cstart - radius):(cend + radius),
    )
    @inbounds for c in cstart:cend
        @simd for r in rstart:rend
            oinput[r, c] = coefx(FT) * (c - 1) + coefy(FT) * (r - 1)
        end
    end
    return OffsetArray(
        AT(input),
        (rstart - radius):(rend + radius),
        (cstart - radius):(cend + radius),
    )
end

# apply the stencil operator
# TODO: ensure this uses FMAs
function stencil(out, weight, radius, input, r, c)
    @inbounds begin
        @simd for ii in (-radius):radius
            out[r, c] += weight[ii, 0] * input[r + ii, c]
        end
        @simd for jj in (-radius):-1
            out[r, c] += weight[0, jj] * input[r, c + jj]
        end
        @simd for jj in 1:radius
            out[r, c] += weight[0, jj] * input[r, c + jj]
        end
    end
end

function stencil_test(
    comms_ctx::ClimaComms.AbstractCommsContext;
    array_types = default_array_types,
    float_types = default_float_types,
    n = default_n,
    radius = default_radius,
    niterations = default_niterations,
    persistent = false,
)
    stencil_size = 4radius + 1

    # initialize and get processor info
    pid, nprocs = ClimaComms.init(comms_ctx)

    # log output only from pid 0
    logger_stream = ClimaComms.iamroot(comms_ctx) ? stderr : devnull
    prev_logger = global_logger(ConsoleLogger(logger_stream, Logging.Info))
    atexit() do
        global_logger(prev_logger)
    end

    # determine 2D grid of pids (closest to square)
    nprocsx, nprocsy = factor(nprocs)
    pidy, pidx = divrem(pid - 1, nprocsx)

    # determine neighbors
    left_nbr = pidx > 0 ? pid - 1 : -1
    right_nbr = pidx < nprocsx - 1 ? pid + 1 : -1
    top_nbr = pidy < nprocsy - 1 ? pid + nprocsx : -1
    bottom_nbr = pidy > 0 ? pid - nprocsx : -1

    # determine dimensions of input and output arrays
    height, rstart, rend = compute_dim(n, nprocsx, pidx)
    width, cstart, cend = compute_dim(n, nprocsy, pidy)
    active_points = (n - 2radius) * (n - 2radius)

    # start tests
    for AT in array_types, FT in float_types
        weight = setup_stencil(AT, FT, radius)
        input = setup_input(
            AT,
            FT,
            height,
            cstart,
            cend,
            width,
            rstart,
            rend,
            radius,
        )
        outa = AT{FT}(undef, height, width)
        fill!(outa, zero(FT))
        out = OffsetArray(outa, rstart:rend, cstart:cend)
        pids = Int[]
        lengths = Int[]
        #---------
        if left_nbr > 0
            push!(pids, left_nbr)
            push!(lengths, radius * width)
        end
        if right_nbr > 0
            push!(pids, right_nbr)
            push!(lengths, radius * width)
        end
        if top_nbr > 0
            push!(pids, top_nbr)
            push!(lengths, height * radius)
        end
        if bottom_nbr > 0
            push!(pids, bottom_nbr)
            push!(lengths, height * radius)
        end
        all_send_buffer = AT{FT}(undef, sum(lengths))
        all_recv_buffer = AT{FT}(undef, sum(lengths))

        graph_ctx = ClimaComms.graph_context(
            comms_ctx,
            all_send_buffer,
            lengths,
            pids,
            all_recv_buffer,
            lengths,
            pids,
            persistent,
        )
        # compute loop
        local_stencil_time = 0
        for iter in 0:niterations
            # start timing after a warmup iteration
            if iter == 1
                ClimaComms.barrier(comms_ctx)
                local_stencil_time = time()
            end
            # fill send buffers
            offset = 0
            if left_nbr > 0
                n = radius * width
                send_buffer = @view all_send_buffer[(offset + 1):(offset + n)]
                send_data =
                    @view input[rstart:(rstart + radius - 1), cstart:cend]
                copyto!(send_buffer, send_data)
                offset += n
            end
            if right_nbr > 0
                n = radius * width
                send_buffer = @view all_send_buffer[(offset + 1):(offset + n)]
                send_data = @view input[(rend - radius + 1):rend, cstart:cend]
                copyto!(send_buffer, send_data)
                offset += n
            end
            if top_nbr > 0
                n = height * radius
                send_buffer = @view all_send_buffer[(offset + 1):(offset + n)]
                send_data = @view input[rstart:rend, (cend - radius + 1):cend]
                copyto!(send_buffer, send_data)
                offset += n
            end
            if bottom_nbr > 0
                n = height * radius
                send_buffer = @view all_send_buffer[(offset + 1):(offset + n)]
                send_data =
                    @view input[rstart:rend, cstart:(cstart + radius - 1)]
                copyto!(send_buffer, send_data)
                offset += n
            end

            # initiate communication
            ClimaComms.start(graph_ctx)

            # apply the stencil operator to the interior while waiting
            # for communication
            @inbounds for c in (cstart + radius):(cend - radius)
                for r in (rstart + radius):(rend - radius)
                    stencil(out, weight, radius, input, r, c)
                end
                ClimaComms.progress(graph_ctx)
            end

            # complete communication
            ClimaComms.finish(graph_ctx)

            # move receive buffers to the input array
            # copy the receive stage buffer for the specified neighbor into `input`

            offset = 0
            if left_nbr > 0
                n = radius * width
                recv_buffer = @view all_recv_buffer[(offset + 1):(offset + n)]
                recv_data =
                    @view input[(rstart - radius):(rstart - 1), cstart:cend]
                copyto!(recv_data, recv_buffer)
                offset += n
            end
            if right_nbr > 0
                n = radius * width
                recv_buffer = @view all_recv_buffer[(offset + 1):(offset + n)]
                recv_data = @view input[(rend + 1):(rend + radius), cstart:cend]
                copyto!(recv_data, recv_buffer)
                offset += n
            end
            if top_nbr > 0
                n = radius * width
                recv_buffer = @view all_recv_buffer[(offset + 1):(offset + n)]
                recv_data = @view input[rstart:rend, (cend + 1):(cend + radius)]
                copyto!(recv_data, recv_buffer)
                offset += n
            end
            if bottom_nbr > 0
                n = radius * width
                recv_buffer = @view all_recv_buffer[(offset + 1):(offset + n)]
                recv_data =
                    @view input[rstart:rend, (cstart - radius):(cstart - 1)]
                copyto!(recv_data, recv_buffer)
                offset += n
            end

            # now we can apply the stencil operator to the exterior
            @inbounds for c in
                          max(cstart, radius + 1):min(cend - radius, n - radius)
                for r in max(rstart, radius + 1):(rstart + radius - 1)
                    stencil(out, weight, radius, input, r, c)
                end
                for r in (rend - radius + 1):min(rend, n - radius)
                    stencil(out, weight, radius, input, r, c)
                end
            end
            @inbounds for c in max(cstart, radius + 1):(cstart + radius - 1)
                for r in max(rstart, radius + 1):(rend - radius)
                    stencil(out, weight, radius, input, r, c)
                end
            end
            @inbounds for c in (cend - radius + 1):min(cend, n - radius)
                for r in max(rstart, radius + 1):(rend - radius)
                    stencil(out, weight, radius, input, r, c)
                end
            end

            # add constant to solution to force refresh of neighbor data, if any
            view(input, rstart:rend, cstart:cend) .+= 1.0
        end

        # get time
        local_stencil_time = (time() - local_stencil_time)
        stencil_time = ClimaComms.reduce(comms_ctx, local_stencil_time, max)

        # compute L1 norm
        local_norm = zero(FT)
        for c in max(cstart, radius + 1):min(cend, n - radius)
            for r in max(rstart, radius + 1):min(rend, n - radius)
                local_norm += abs(out[r, c])
            end
        end
        norm = ClimaComms.reduce(comms_ctx, local_norm, +)

        # verify correctness
        if ClimaComms.iamroot(comms_ctx)
            norm /= active_points
            refnorm = (niterations + 1) * (coefx(FT) + coefy(FT))
            if abs(norm - refnorm) > epsilon(FT)
                @error "$AT{$FT}: norm = $(norm), reference = $(refnorm)\n"
                # TODO: uncomment this when ClimaCommsSA is fixed
                # see: https://github.com/CliMA/ClimaComms.jl/issues/6
                #ClimaComms.abort(comms_ctx, -1)
            end

            # flops/stencil: 2 flops (fma) for each point in the stencil
            # plus one flop for the update of the input array
            flops = (2 * stencil_size + 1) * active_points
            avgtime = stencil_time / niterations
            @info @sprintf "%f MFlops/s\n" 1.0e-6 * flops / avgtime
        end

        gathered = ClimaComms.gather(
            comms_ctx,
            fill(ClimaComms.mypid(comms_ctx), (3, 3)),
        )
        if ClimaComms.iamroot(comms_ctx)
            @test gathered == repeat(
                reshape(1:ClimaComms.nprocs(comms_ctx), (1, :)),
                inner = (3, 3),
            )
        else
            @test isnothing(gathered)
        end
        # test for allreduce!
        sendrecvbuf = [pid]
        ClimaComms.allreduce!(comms_ctx, sendrecvbuf, +)
        @test sendrecvbuf == [div(nprocs * (nprocs + 1), 2)]

        sendbuf = [pid]
        recvbuf = [0]
        ClimaComms.allreduce!(comms_ctx, sendbuf, recvbuf, +)
        @test recvbuf == [div(nprocs * (nprocs + 1), 2)]
        # test for allreduce
        sendbuf = pid
        recvbuf = ClimaComms.allreduce(comms_ctx, sendbuf, +)
        @test recvbuf == div(nprocs * (nprocs + 1), 2)
    end

    return nothing
end
