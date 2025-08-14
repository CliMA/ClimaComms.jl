using ClimaComms

ClimaComms.@import_required_backends
context = ClimaComms.context()
pid, nprocs = ClimaComms.init(context)
device = ClimaComms.device(context)
AT = ClimaComms.array_type(device)

x_max = 100
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

Base.@propagate_inbounds function nth_deriv_along_axis1(array, n, i, indices...)
    @assert n >= 0
    n == 0 && return array[i, indices...]
    prev_i = i == 1 ? size(array, 1) : i - 1
    next_i = i == size(array, 1) ? 1 : i + 1
    value_at_prev_i = nth_deriv_along_axis1(array, n - 1, prev_i, indices...)
    value_at_next_i = nth_deriv_along_axis1(array, n - 1, next_i, indices...)
    value_at_i = nth_deriv_along_axis1(array, n - 1, i, indices...)
    return (value_at_next_i - 2 * value_at_i + value_at_prev_i) / 2
end

ref_array_copy!(a_copy, a, device) = ClimaComms.@cuda_sync device a_copy .= a
threaded_array_copy!(a_copy, a, device) = ClimaComms.@cuda_sync device begin
    ClimaComms.@threaded device for y in axes(a, 2), x in axes(a, 1)
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
threaded_array_max!(a_max, a, device) = ClimaComms.@cuda_sync device begin
    ClimaComms.@threaded device begin
        ClimaComms.@shmem_threaded synced_x_iterator = axes(a, 1)
        for y in axes(a, 2)
            T = eltype(a)
            N = length(synced_x_iterator)
            a_col = ClimaComms.shmem_array(device, T, N)
            for x in synced_x_iterator
                @inbounds a_col[x] = a[x, y]
            end
            a_col_max = ClimaComms.shmem_maximum!(ClimaComms.unsync(a_col))
            ClimaComms.@unique_shmem_thread synced_x_iterator begin
                @inbounds a_max[y] = a_col_max
            end
        end
    end
end

println()
@info "Benchmarking max along 1st axis of $x_max×$y_max matrix:"
a_max_ref = a[1, :]
a_max_threaded = a[1, :]
reads_and_writes_for_max = sizeof(a) + sizeof(a_max_ref)
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
        nth_deriv_at(index) = nth_deriv_along_axis1(a, n, index[1], index[2])
        ∂ⁿa∂xⁿ .= nth_deriv_at.(cartesian_indices)
    end
threaded_deriv_1!(∂ⁿa∂xⁿ, a, device, n) = ClimaComms.@cuda_sync device begin
    ClimaComms.@threaded device for y in axes(a, 2), x in axes(a, 1)
        @inbounds ∂ⁿa∂xⁿ[x, y] = nth_deriv_along_axis1(a, n, x, y)
    end
end
threaded_deriv_2!(∂ⁿa∂xⁿ, a, device, n) = ClimaComms.@cuda_sync device begin
    ClimaComms.@threaded device begin
        ClimaComms.@shmem_threaded synced_x_iterator = axes(a, 1)
        for y in axes(a, 2)
            for x in ClimaComms.unsync(synced_x_iterator)
                @inbounds ∂ⁿa∂xⁿ[x, y] = nth_deriv_along_axis1(a, n, x, y)
            end
        end
    end
end
threaded_deriv_3!(∂ⁿa∂xⁿ, a, device, n) = ClimaComms.@cuda_sync device begin
    ClimaComms.@threaded device begin
        ClimaComms.@shmem_threaded synced_x_iterator = axes(a, 1)
        for y in axes(a, 2)
            T = eltype(a)
            N = length(synced_x_iterator)
            a_col = ClimaComms.shmem_array(device, T, N)
            for x in synced_x_iterator
                @inbounds a_col[x] = a[x, y]
            end
            for x in ClimaComms.unsync(synced_x_iterator)
                @inbounds ∂ⁿa∂xⁿ[x, y] = nth_deriv_along_axis1(a_col, n, x)
            end
        end
    end
end
threaded_deriv_4!(∂ⁿa∂xⁿ, a, device, n) = ClimaComms.@cuda_sync device begin
    ClimaComms.@threaded device begin
        ClimaComms.@shmem_threaded synced_x_iterator = axes(a, 1)
        for y in axes(a, 2)
            T = eltype(a)
            N = length(synced_x_iterator)
            a_col = ClimaComms.shmem_array(device, T, N)
            for x in synced_x_iterator
                @inbounds a_col[x] = a[x, y]
            end
            m = n ÷ 2
            ∂ᵐa∂xᵐ_col = ClimaComms.shmem_array(device, T, N)
            for x in synced_x_iterator
                @inbounds ∂ᵐa∂xᵐ_col[x] = nth_deriv_along_axis1(a_col, m, x)
            end
            for x in ClimaComms.unsync(synced_x_iterator)
                @inbounds ∂ⁿa∂xⁿ[x, y] =
                    nth_deriv_along_axis1(∂ᵐa∂xᵐ_col, n - m, x)
            end
        end
    end
end
threaded_deriv_5!(∂ⁿa∂xⁿ, a, device, n) = ClimaComms.@cuda_sync device begin
    ClimaComms.@threaded device begin
        ClimaComms.@shmem_threaded synced_x_iterator = axes(a, 1)
        for y in axes(a, 2)
            if n == 0
                for x in ClimaComms.unsync(synced_x_iterator)
                    @inbounds ∂ⁿa∂xⁿ[x, y] = a[x, y]
                end
            else
                T = eltype(a)
                N = length(synced_x_iterator)
                two_cols = ClimaComms.shmem_array(device, T, N, 2)
                for x in synced_x_iterator
                    @inbounds two_cols[x, 1] = a[x, y]
                end
                prev_col_idx = 1
                for m in 1:(n - 1)
                    for x in synced_x_iterator
                        @inbounds two_cols[x, 3 - prev_col_idx] =
                            nth_deriv_along_axis1(two_cols, 1, x, prev_col_idx)
                    end
                    prev_col_idx = 3 - prev_col_idx
                end
                for x in ClimaComms.unsync(synced_x_iterator)
                    @inbounds ∂ⁿa∂xⁿ[x, y] =
                        nth_deriv_along_axis1(two_cols, 1, x, prev_col_idx)
                end
            end
        end
    end
end

println()
@info "Benchmarking n-th derivative along 1st axis of $x_max×$y_max matrix:"
∂ⁿa∂xⁿ_ref = similar(a)
∂ⁿa∂xⁿ_threaded = similar(a)
reads_and_writes_for_deriv = sizeof(a) + sizeof(∂ⁿa∂xⁿ_ref)
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
        nth_deriv_at(index) = nth_deriv_along_axis1(a, n, index[1], index[2])
        ∂ⁿa∂xⁿ .= nth_deriv_at.(cartesian_indices)
        maximum!(reshape(∂ⁿa∂xⁿ_max, 1, :), ∂ⁿa∂xⁿ)
    end
threaded_deriv_max_1!(∂ⁿa∂xⁿ_max, a, device, n) =
    ClimaComms.@cuda_sync device begin
        ClimaComms.@threaded device begin
            ClimaComms.@shmem_threaded synced_x_iterator = axes(a, 1)
            for y in axes(a, 2)
                T = eltype(a)
                N = length(synced_x_iterator)
                a_col = ClimaComms.shmem_array(device, T, N)
                for x in synced_x_iterator
                    @inbounds a_col[x] = a[x, y]
                end
                ∂ⁿa∂xⁿ_col = ClimaComms.shmem_array(device, T, N)
                for x in synced_x_iterator
                    @inbounds ∂ⁿa∂xⁿ_col[x] = nth_deriv_along_axis1(a_col, n, x)
                end
                ∂ⁿa∂xⁿ_col_max =
                    ClimaComms.shmem_maximum!(ClimaComms.unsync(∂ⁿa∂xⁿ_col))
                ClimaComms.@unique_shmem_thread synced_x_iterator begin
                    @inbounds ∂ⁿa∂xⁿ_max[y] = ∂ⁿa∂xⁿ_col_max
                end
            end
        end
    end
threaded_deriv_max_2!(∂ⁿa∂xⁿ_max, a, device, n) =
    ClimaComms.@cuda_sync device begin
        ClimaComms.@threaded device begin
            ClimaComms.@shmem_threaded synced_x_iterator = axes(a, 1)
            for y in axes(a, 2)
                T = eltype(a)
                N = length(synced_x_iterator)
                two_cols = ClimaComms.shmem_array(device, T, N, 2)
                for x in synced_x_iterator
                    @inbounds two_cols[x, 1] = a[x, y]
                end
                prev_col_idx = 1
                for m in 1:n
                    for x in synced_x_iterator
                        @inbounds two_cols[x, 3 - prev_col_idx] =
                            nth_deriv_along_axis1(two_cols, 1, x, prev_col_idx)
                    end
                    prev_col_idx = 3 - prev_col_idx
                end
                @inbounds ∂ⁿa∂xⁿ_col = view(two_cols, :, prev_col_idx)
                ∂ⁿa∂xⁿ_col_max =
                    ClimaComms.shmem_maximum!(ClimaComms.unsync(∂ⁿa∂xⁿ_col))
                ClimaComms.@unique_shmem_thread synced_x_iterator begin
                    @inbounds ∂ⁿa∂xⁿ_max[y] = ∂ⁿa∂xⁿ_col_max
                end
            end
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
