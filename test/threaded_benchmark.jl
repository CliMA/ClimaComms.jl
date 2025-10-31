using ClimaComms
using ClimaComms: StaticArrays

ClimaComms.@import_required_backends
context = ClimaComms.context()
pid, nprocs = ClimaComms.init(context)
device = ClimaComms.device(context)
AT = ClimaComms.array_type(device)

const x_max = 100
y_max = device isa ClimaComms.CUDADevice ? 1000000 : 10000
a = AT(rand(x_max, y_max))
cartesian_indices = AT(CartesianIndices(a))

macro min_elapsed(expr)
    return quote
        min_elapsed_time = Inf
        start_time_ns = time_ns()
        while time_ns() - start_time_ns < 1e9 # Benchmark for 1 second.
            min_elapsed_time = min(min_elapsed_time, @elapsed $(esc(expr)))
        end
        min_elapsed_time
    end
end

print_latency(latency) =
    if latency < 1
        @info "    Latency = $(round(1000 * latency; sigdigits = 3)) ms"
    else
        @info "    Latency = $(round(latency; sigdigits = 3)) s"
    end

function print_time_and_bandwidth(device, reads_and_writes, time)
    if time < 1
        @info "    Time = $(round(1000 * time; sigdigits = 3)) ms"
    else
        @info "    Time = $(round(time; sigdigits = 3)) s"
    end
    bandwidth_gbps = (reads_and_writes / 1024^3) / time
    if device isa ClimaComms.CUDADevice
        cuda_device = CUDA.device()
        memory_bits_attr = CUDA.DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH
        memory_kilohertz_attr = CUDA.DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE
        # Multiply memory bus width by 2 for high bandwidth memory (HBM).
        peak_bandwidth_gbps =
            (2 * CUDA.attribute(cuda_device, memory_bits_attr) / (8 * 1024^3)) *
            (CUDA.attribute(cuda_device, memory_kilohertz_attr) * 1000)
        bandwidth_percent = 100 * bandwidth_gbps / peak_bandwidth_gbps
        @info "    Bandwidth = $(round(bandwidth_percent; sigdigits = 3))% of \
                   $(round(peak_bandwidth_gbps; sigdigits = 3)) GB/s"
    else
        @info "    Bandwidth = $(round(bandwidth_gbps; sigdigits = 3)) GB/s"
    end
end

Base.@propagate_inbounds function nth_deriv_along_x_axis(array, n, x, y...)
    @assert n >= 0
    n == 0 && return array[x, y...]
    prev_x = x == 1 ? size(array, 1) : x - 1
    next_x = x == size(array, 1) ? 1 : x + 1
    value_at_prev_x = nth_deriv_along_x_axis(array, n - 1, prev_x, y...)
    value_at_next_x = nth_deriv_along_x_axis(array, n - 1, next_x, y...)
    value_at_x = nth_deriv_along_x_axis(array, n - 1, x, y...)
    return (value_at_next_x - 2 * value_at_x + value_at_prev_x) / 2
end
Base.@propagate_inbounds nth_deriv_along_x_axis(array, n, i::CartesianIndex) =
    nth_deriv_along_x_axis(array, n, Tuple(i)...)
Base.@propagate_inbounds nth_deriv_along_x_axis(a, n) =
    Base.@propagate_inbounds (is...,) -> nth_deriv_along_x_axis(a, n, is...)
Base.@propagate_inbounds deriv_along_x_axis(a) = nth_deriv_along_x_axis(a, 1)

ref_array_copy!(a_copy, a, device) = ClimaComms.@cuda_sync device a_copy .= a
threaded_array_copy!(a_copy, a, device) = ClimaComms.@threaded device begin
    for y in axes(a, 2), x in axes(a, 1)
        @inbounds a_copy[x, y] = a[x, y]
    end
end

println()
@info "Benchmarking copy of $x_max×$y_max matrix:"
a_copy_ref = similar(a)
a_copy_threaded = similar(a)
reads_and_writes_for_copy = sizeof(a) + sizeof(a_copy_ref)
for (array_copy!, info_string, a_copy) in (
    (ref_array_copy!, "reference copy", a_copy_ref),
    (threaded_array_copy!, "@threaded copy", a_copy_threaded),
)
    time_0 = @elapsed array_copy!(a_copy, a, device)
    time = @min_elapsed array_copy!(a_copy, a, device)
    @info info_string
    print_latency(time_0 - time)
    print_time_and_bandwidth(device, reads_and_writes_for_copy, time)
end
@test a_copy_threaded == a_copy_ref

ref_array_max!(a_max, a, device) =
    ClimaComms.@cuda_sync device maximum!(reshape(a_max, 1, :), a)
threaded_array_max!(a_max, a, device, debug) =
    ClimaComms.@threaded device debug=debug begin
        ClimaComms.@shmem_threaded X = StaticArrays.SOneTo(x_max)
        for y in axes(a, 2)
            a_col = ClimaComms.shmem_array(X, eltype(a))
            @inbounds a_col .= view(a, :, y)
            @inbounds a_col_max = view(a_max, y)
            ClimaComms.shmem_maximum!(a_col_max, a_col)
        end
    end
threaded_array_max!(a_max, a, device) =
    threaded_array_max!(a_max, a, device, nothing)

println()
@info "Benchmarking max along 1st axis of $x_max×$y_max matrix:"
a_max_ref = a[1, :]
a_max_threaded = a[1, :]
reads_and_writes_for_max = sizeof(a) + sizeof(a_max_ref)
for (debug_sym, debug_str) in ((:warn, "Julia"), (:llvm, "LLVM"), (:ptx, "PTX"))
    println()
    @info "$debug_str code for max along 1st axis of matrix:"
    threaded_array_max!(a_max_threaded, a, device, Val(debug_sym))
end
for (array_max!, info_string, a_max) in (
    (ref_array_max!, "reference max", a_max_ref),
    (threaded_array_max!, "@threaded max", a_max_threaded),
)
    time_0 = @elapsed array_max!(a_max, a, device)
    time = @min_elapsed array_max!(a_max, a, device)
    @info info_string
    print_latency(time_0 - time)
    print_time_and_bandwidth(device, reads_and_writes_for_max, time)
end
@test a_max_threaded == a_max_ref

ref_deriv!(∂ⁿa∂xⁿ, a, device, n, cartesian_indices) =
    ClimaComms.@cuda_sync device begin
        ∂ⁿa∂xⁿ .= nth_deriv_along_x_axis(a, n).(cartesian_indices)
    end
threaded_deriv_1!(∂ⁿa∂xⁿ, a, device, n) =
    ClimaComms.@threaded device for y in axes(a, 2), x in axes(a, 1)
        @inbounds ∂ⁿa∂xⁿ[x, y] = nth_deriv_along_x_axis(a, n, x, y)
    end
threaded_deriv_2!(∂ⁿa∂xⁿ, a, device, n, debug) =
    ClimaComms.@threaded device debug=debug begin
        ClimaComms.@shmem_threaded X = StaticArrays.SOneTo(x_max)
        for y in axes(a, 2)
            @inbounds view(∂ⁿa∂xⁿ, :, y) .= nth_deriv_along_x_axis(a, n).(X, y)
        end
    end
threaded_deriv_2!(∂ⁿa∂xⁿ, a, device, n) =
    threaded_deriv_2!(∂ⁿa∂xⁿ, a, device, n, nothing)
threaded_deriv_3!(∂ⁿa∂xⁿ, a, device, n) = ClimaComms.@threaded device begin
    ClimaComms.@shmem_threaded X = StaticArrays.SOneTo(x_max)
    for y in axes(a, 2)
        a_col = ClimaComms.shmem_array(X, eltype(a))
        @inbounds a_col .= view(a, :, y)
        @inbounds view(∂ⁿa∂xⁿ, :, y) .= nth_deriv_along_x_axis(a_col, n).(X)
    end
end
threaded_deriv_4!(∂ⁿa∂xⁿ, a, device, n) = ClimaComms.@threaded device begin
    ClimaComms.@shmem_threaded X = StaticArrays.SOneTo(x_max)
    for y in axes(a, 2)
        a_col = ClimaComms.shmem_array(X, eltype(a))
        @inbounds a_col .= view(a, :, y)
        m = n ÷ 2
        @inbounds ∂ᵐa∂xᵐ_col = nth_deriv_along_x_axis(a_col, m).(X)
        @inbounds view(∂ⁿa∂xⁿ, :, y) .=
            nth_deriv_along_x_axis(∂ᵐa∂xᵐ_col, n - m).(X)
    end
end
threaded_deriv_5!(∂ⁿa∂xⁿ, a, device, n) = ClimaComms.@threaded device begin
    ClimaComms.@shmem_threaded X = StaticArrays.SOneTo(x_max)
    for y in axes(a, 2)
        if n == 0
            @inbounds view(∂ⁿa∂xⁿ, :, y) .= view(a, :, y)
            continue
        end
        two_cols = ClimaComms.shmem_array(X, eltype(a), (length(X), 2))
        @inbounds view(two_cols, :, 1) .= view(a, :, y)
        prev_col_index = 1
        for m in 1:(n - 1)
            new_col_index = 3 - prev_col_index
            @inbounds view(two_cols, :, new_col_index) .=
                deriv_along_x_axis(two_cols).(X, prev_col_index)
            prev_col_index = new_col_index
        end
        @inbounds view(∂ⁿa∂xⁿ, :, y) .=
            deriv_along_x_axis(two_cols).(X, prev_col_index)
    end
end

println()
@info "Benchmarking n-th derivative along 1st axis of $x_max×$y_max matrix:"
∂ⁿa∂xⁿ_ref = similar(a)
∂ⁿa∂xⁿ_threaded = similar(a)
reads_and_writes_for_deriv = sizeof(a) + sizeof(∂ⁿa∂xⁿ_ref)
for (debug_sym, debug_str) in ((:warn, "Julia"), (:llvm, "LLVM"), (:ptx, "PTX"))
    println()
    @info "$debug_str code for 0th derivative along 1st axis of matrix:"
    threaded_deriv_2!(∂ⁿa∂xⁿ_threaded, a, device, 0, Val(debug_sym))
end
for n in (0, 2, 4, 6)
    time_0 =
        n > 0 ? 0 :
        @elapsed ref_deriv!(∂ⁿa∂xⁿ_ref, a, device, n, cartesian_indices)
    time = @min_elapsed ref_deriv!(∂ⁿa∂xⁿ_ref, a, device, n, cartesian_indices)
    @info "reference derivative, n = $n (using Cartesian index matrix):"
    n == 0 && print_latency(time_0 - time)
    print_time_and_bandwidth(device, reads_and_writes_for_deriv, time)
    for (threaded_deriv!, shmem_string) in (
        (threaded_deriv_1!, "both axes distributed across all threads"),
        (threaded_deriv_2!, "1st axis distributed within GPU blocks"),
        (threaded_deriv_3!, "shmem along 1st axis, 1 synchronization"),
        (threaded_deriv_4!, "shmem along 1st axis, 2 synchronizations"),
        (threaded_deriv_5!, "shmem along 1st axis, n synchronizations"),
    )
        time_0 =
            n > 0 ? 0 : @elapsed threaded_deriv!(∂ⁿa∂xⁿ_threaded, a, device, n)
        time = @min_elapsed threaded_deriv!(∂ⁿa∂xⁿ_threaded, a, device, n)
        @info "@threaded derivative, n = $n ($shmem_string):"
        n == 0 && print_latency(time_0 - time)
        print_time_and_bandwidth(device, reads_and_writes_for_deriv, time)
        @test ∂ⁿa∂xⁿ_threaded == ∂ⁿa∂xⁿ_ref
    end
end
for n in (8, 12, 25, 50, 100, 200, 400)
    for (threaded_deriv!, shmem_string) in (
        (threaded_deriv_4!, "shmem along 1st axis, 2 synchronizations"),
        (threaded_deriv_5!, "shmem along 1st axis, n synchronizations"),
    )
        n > 12 && endswith(shmem_string, "2 synchronizations") && continue
        time = @min_elapsed threaded_deriv!(∂ⁿa∂xⁿ_threaded, a, device, n)
        @info "@threaded derivative, n = $n ($shmem_string):"
        print_time_and_bandwidth(device, reads_and_writes_for_deriv, time)
    end
end

ref_deriv_max!(∂ⁿa∂xⁿ_max, ∂ⁿa∂xⁿ, a, device, n, cartesian_indices) =
    ClimaComms.@cuda_sync device begin
        ∂ⁿa∂xⁿ .= nth_deriv_along_x_axis(a, n).(cartesian_indices)
        maximum!(reshape(∂ⁿa∂xⁿ_max, 1, :), ∂ⁿa∂xⁿ)
    end
threaded_deriv_max_1!(∂ⁿa∂xⁿ_max, a, device, n) =
    ClimaComms.@threaded device begin
        ClimaComms.@shmem_threaded X = StaticArrays.SOneTo(x_max)
        for y in axes(a, 2)
            a_col = ClimaComms.shmem_array(X, eltype(a))
            @inbounds a_col .= view(a, :, y)
            @inbounds ∂ⁿa∂xⁿ_col = nth_deriv_along_x_axis(a_col, n).(X)
            @inbounds ∂ⁿa∂xⁿ_col_max = view(∂ⁿa∂xⁿ_max, y)
            ClimaComms.shmem_maximum!(∂ⁿa∂xⁿ_col_max, ∂ⁿa∂xⁿ_col)
        end
    end
threaded_deriv_max_2!(∂ⁿa∂xⁿ_max, a, device, n) =
    ClimaComms.@threaded device begin
        ClimaComms.@shmem_threaded X = StaticArrays.SOneTo(x_max)
        for y in axes(a, 2)
            two_cols = ClimaComms.shmem_array(X, eltype(a), (length(X), 2))
            @inbounds view(two_cols, :, 1) .= view(a, :, y)
            prev_col_index = 1
            for m in 1:n
                new_col_index = 3 - prev_col_index
                @inbounds view(two_cols, :, new_col_index) .=
                    deriv_along_x_axis(two_cols).(X, prev_col_index)
                prev_col_index = new_col_index
            end
            @inbounds ∂ⁿa∂xⁿ_col = view(two_cols, :, prev_col_index)
            @inbounds ∂ⁿa∂xⁿ_col_max = view(∂ⁿa∂xⁿ_max, y)
            ClimaComms.shmem_maximum!(∂ⁿa∂xⁿ_col_max, ∂ⁿa∂xⁿ_col)
        end
    end

println()
@info "Benchmarking max of n-th derivative along 1st axis of $x_max×$y_max matrix:"
∂ⁿa∂xⁿ = similar(a)
∂ⁿa∂xⁿ_ref_max = a[1, :]
∂ⁿa∂xⁿ_threaded_max = a[1, :]
reads_and_writes_for_deriv_max = sizeof(a) + sizeof(∂ⁿa∂xⁿ_ref_max)
for n in (0, 2, 4, 6)
    time_0 =
        n > 0 ? 0 :
        @elapsed ref_deriv_max!(
            ∂ⁿa∂xⁿ_ref_max,
            ∂ⁿa∂xⁿ,
            a,
            device,
            n,
            cartesian_indices,
        )
    time = @min_elapsed ref_deriv_max!(
        ∂ⁿa∂xⁿ_ref_max,
        ∂ⁿa∂xⁿ,
        a,
        device,
        n,
        cartesian_indices,
    )
    @info "reference max of derivative, n = $n (using Cartesian index matrix):"
    n == 0 && print_latency(time_0 - time)
    print_time_and_bandwidth(device, reads_and_writes_for_deriv_max, time)
    for (threaded_deriv_max!, shmem_string) in (
        (threaded_deriv_max_1!, "1 + log2(x_max / 32) synchronizations"),
        (threaded_deriv_max_2!, "n + log2(x_max / 32) synchronizations"),
    )
        time_0 =
            n > 0 ? 0 :
            @elapsed threaded_deriv_max!(∂ⁿa∂xⁿ_threaded_max, a, device, n)
        time =
            @min_elapsed threaded_deriv_max!(∂ⁿa∂xⁿ_threaded_max, a, device, n)
        @info "@threaded max of derivative, n = $n ($shmem_string):"
        n == 0 && print_latency(time_0 - time)
        print_time_and_bandwidth(device, reads_and_writes_for_deriv_max, time)
        @test ∂ⁿa∂xⁿ_threaded_max == ∂ⁿa∂xⁿ_ref_max
    end
end

# TODO: Organize all of these measurements into a table.
