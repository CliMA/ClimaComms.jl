push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using CUDA
using Logging
using MPI
using OffsetArrays
using Printf

using ClimaComms

# configuration: grid dimension, stencil radius, etc.
const n = 256
const radius = 2
const niterations = 50
const stencil_size = 4radius + 1

# floating point precisions to test with
const float_types = (Float64, Float32)
epsilon(::Type{Float64}) = 1.0e-8
epsilon(::Type{Float32}) = 0.0001f0
coefx(::Type{Float64}) = 1.0
coefx(::Type{Float32}) = 1.0f0
coefy(::Type{Float64}) = 1.0
coefy(::Type{Float32}) = 1.0f0

# array types to test with
const array_types = CUDA.has_cuda() ? (CuArray, Array) : (Array,)

# initialize and get rank info
const rank, nranks = ClimaComms.init(ClimaComms.MPICommsType)

# log output only from rank 0
logger_stream = rank == 0 ? stderr : devnull
prev_logger = global_logger(ConsoleLogger(logger_stream, Logging.Info))
atexit() do
    global_logger(prev_logger)
end

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

# determine 2D grid of ranks (closest to square)
const nranksx, nranksy = factor(nranks)
const ranky, rankx = divrem(rank, nranksx)

# determine neighbors
const left_nbr = rankx > 0 ? rank - 1 : -1
const right_nbr = rankx < nranksx - 1 ? rank + 1 : -1
const top_nbr = ranky < nranksy - 1 ? rank + nranksx : -1
const bottom_nbr = ranky > 0 ? rank - nranksx : -1

# partition `n` into `nr` parts and return the size and the offsets
# of rank `rdim`s part
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

# determine dimensions of input and output arrays
const height, rstart, rend = compute_dim(n, nranksx, rankx)
const width, cstart, cend = compute_dim(n, nranksy, ranky)
const active_points = (n - 2radius) * (n - 2radius)

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

# neighbor information
struct Neighbor
    rank::Int
    send_buf::Union{ClimaComms.RelayBuffer, Missing}
    recv_buf::Union{ClimaComms.RelayBuffer, Missing}
    function Neighbor(rank, AT, FT, dims...)
        if rank >= 0
            send_buf = ClimaComms.RelayBuffer{FT}(AT, missing, dims...)
            recv_buf = ClimaComms.RelayBuffer{FT}(AT, missing, dims...)
        else
            send_buf = recv_buf = missing
        end
        new(rank, send_buf, recv_buf)
    end
end

# fill the send stage buffer for the specified neighbor from `input`
function fill_send_buf(nbr, input, rrange, crange)
    if nbr.rank >= 0
        send_buf = ClimaComms.get_stage(nbr.send_buf)
        send_buf .= view(input, rrange, crange)
    end
end

# copy the receive stage buffer for the specified neighbor into `input`
function empty_recv_buf(nbr, input, rrange, crange)
    if nbr.rank >= 0
        recv_buf = ClimaComms.get_stage(nbr.recv_buf)
        view(input, rrange, crange) .= recv_buf
    end
end

# apply the stencil operator
# TODO: ensure this uses FMAs
function stencil(out, weight, input, r, c)
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

    # set up communication buffers for neighbors
    neighbors = (;
        left = Neighbor(left_nbr, AT, FT, radius, width),
        right = Neighbor(right_nbr, AT, FT, radius, width),
        top = Neighbor(top_nbr, AT, FT, height, radius),
        bottom = Neighbor(bottom_nbr, AT, FT, height, radius),
    )

    # create ClimaComms context
    comms_ctx = ClimaComms.MPICommsType(
        filter(r -> r >= 0, [nbr.rank for nbr in neighbors]),
        filter(b -> !ismissing(b), [nbr.recv_buf for nbr in neighbors]),
        filter(b -> !ismissing(b), [nbr.send_buf for nbr in neighbors]),
    )

    # compute loop
    local_stencil_time = 0
    for iter in 0:niterations
        # start timing after a warmup iteration
        if iter == 1
            MPI.Barrier(MPI.COMM_WORLD)
            local_stencil_time = time()
        end

        # fill send buffers
        fill_send_buf(
            neighbors.left,
            input,
            rstart:rstart + radius - 1,
            cstart:cend,
        )
        fill_send_buf(
            neighbors.right,
            input,
            rend - radius + 1:rend,
            cstart:cend,
        )
        fill_send_buf(
            neighbors.top,
            input,
            rstart:rend,
            cend - radius + 1:cend,
        )
        fill_send_buf(
            neighbors.bottom,
            input,
            rstart:rend,
            cstart:cstart + radius - 1,
        )

        # initiate communication
        ClimaComms.start(comms_ctx)

        # apply the stencil operator to the interior while waiting
        # for communication
        @inbounds for c in (cstart + radius):(cend - radius)
            for r in (rstart + radius):(rend - radius)
                stencil(out, weight, input, r, c)
            end
            ClimaComms.progress(comms_ctx)
        end

        # complete communication
        ClimaComms.finish(comms_ctx)

        # move receive buffers to the input array
        empty_recv_buf(
            neighbors.left,
            input,
            rstart - radius:rstart - 1,
            cstart:cend,
        )
        empty_recv_buf(
            neighbors.right,
            input,
            rend + 1:rend + radius,
            cstart:cend,
        )
        empty_recv_buf(
            neighbors.top,
            input,
            rstart:rend,
            cend + 1:cend + radius,
        )
        empty_recv_buf(
            neighbors.bottom,
            input,
            rstart:rend,
            cstart - radius:cstart - 1,
        )

        # now we can apply the stencil operator to the exterior
        @inbounds for c in max(cstart, radius + 1):min(cend - radius, n - radius)
            for r in max(rstart, radius + 1):rstart + radius - 1
                stencil(out, weight, input, r, c)
            end
            for r in rend - radius + 1:min(rend, n - radius)
                stencil(out, weight, input, r, c)
            end
        end
        @inbounds for c in max(cstart, radius + 1):cstart + radius - 1
            for r in max(rstart, radius + 1):rend - radius
                stencil(out, weight, input, r, c)
            end
        end
        @inbounds for c in cend - radius + 1:min(cend, n - radius)
            for r in max(rstart, radius + 1):rend - radius
                stencil(out, weight, input, r, c)
            end
        end

        # add constant to solution to force refresh of neighbor data, if any
        view(input, rstart:rend, cstart:cend) .+= 1.0
    end

    # get time
    local_stencil_time = (time() - local_stencil_time)
    stencil_time =
        MPI.Reduce(local_stencil_time, MPI.MAX, 0, MPI.COMM_WORLD)

    # compute L1 norm
    local_norm = zero(FT)
    @inbounds for c in max(cstart, radius + 1):min(cend, n - radius)
        for r in max(rstart, radius + 1):min(rend, n - radius)
            local_norm += abs(out[r, c])
        end
    end
    norm = MPI.Reduce(local_norm, +, 0, MPI.COMM_WORLD)

    # verify correctness
    if rank == 0
        norm /= active_points
        refnorm = (niterations + 1) * (coefx(FT) + coefy(FT))
        if abs(norm - refnorm) > epsilon(FT)
            @error "$AT{$FT}: norm = $(norm), reference = $(refnorm)\n"
            MPI.Abort(MPI.COMM_WORLD, -1)
        end

        # flops/stencil: 2 flops (fma) for each point in the stencil
        # plus one flop for the update of the input array
        flops = (2 * stencil_size + 1) * active_points
        avgtime = stencil_time / niterations
        @info @sprintf "%f MFlops/s\n" 1.0e-6 * flops / avgtime
    end
end
