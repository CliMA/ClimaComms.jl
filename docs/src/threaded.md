# Writing `@threaded` Kernels

**This documentation needs to be updated for the new API.**

In this section, we will give a tutorial for writing device-agnostic kernels
using `ClimaComms.@threaded`. As an illustrative example, we will show how to
approximate the `n`-th derivative of a matrix in an efficient GPU kernel.

We can compute the first derivative of an `array` along each column using
second-order finite differences with periodic boundary conditions as

```julia
Base.@propagate_inbounds function axis1_deriv(array, index1, indices...)
    prev_index1 = index1 == 1 ? size(array, 1) : index1 - 1
    next_index1 = index1 == size(array, 1) ? 1 : index1 + 1
    return (
        array[next_index1, indices...] -
        2 * array[index1, indices...] +
        array[prev_index1, indices...]
    ) / 2
end
```

This can be recursively extended to the `n`-th derivative as
```julia
Base.@propagate_inbounds function axis1_nth_deriv(n, array, index1, indices...)
    prev_index1 = index1 == 1 ? size(array, 1) : index1 - 1
    next_index1 = index1 == size(array, 1) ? 1 : index1 + 1
    if n == 1
        return (
            array[next_index1, indices...] -
            2 * array[index1, indices...] +
            array[prev_index1, indices...]
        ) / 2
    else
        return (
            axis1_nth_deriv(n - 1, array, next_index1, indices...) -
            2 * axis1_nth_deriv(n - 1, array, index1, indices...) +
            axis1_nth_deriv(n - 1, array, prev_index1, indices...)
        ) / 2
    end
end
```

The simplest way to parallelize this function over the indices of a `matrix` is
to directly evaluate the function at every point:
```julia
columnwise_nth_deriv!(result, matrix, n, device) =
    ClimaComms.@threaded device for i in axes(matrix, 1), j in axes(matrix, 2)
        @inbounds result[i, j] = axis1_nth_deriv(n, matrix, i, j)
    end
```

However, this implementation results in a very large number of global memory
reads: `matrix` is accessed `O(3^n)` times, which becomes prohibitively large as
`n` grows. For example, here is the performance for a `100 × 1000` matrix with
`n = 1` and `n = 10`:
```julia
# TODO: Add performance benchmarks.
```

To reduce the number of global memory reads, we can move each column of the
`matrix` into shared memory before using it to evaluate the `n`-th derivative.
This makes each row index `i in axes(matrix, 1)` interdependent with every other
row index, which we need to explicitly declare:
```julia
columnwise_nth_deriv!(result, matrix, n, device, ::Val{N_rows}) where {N_rows} =
    ClimaComms.@threaded device begin
        for i in @interdependent(axes(matrix, 1)), j in axes(matrix, 2)
            T = eltype(matrix)
            matrix_col =
                ClimaComms.static_shared_memory_array(device, T, N_rows)
            @inbounds begin
                ClimaComms.@sync_interdependent matrix_col[i] = matrix[i, j]
                ClimaComms.@sync_interdependent result[i, j] =
                    axis1_nth_deriv(n, matrix_col, i)
            end
        end
    end
```
The shared memory array used to store each column is statically-sized, so the
number of rows in each column must be passed as a static parameter. Also, every
use of the interdependent variable `i` must occur inside a
`@sync_interdependent` expression.

In this implementation, there are `O(N_rows)` global memory reads and
`O(N_rows * 3^n)` shared memory reads, which improves the performance:
```julia
# TODO: Add performance benchmarks.
```

We can further improve runtime by splitting the `O(N_rows * 3^n)` shared memory
reads into two sets of `O(N_rows * 3^(n ÷ 2))` shared memory reads:
```julia
columnwise_nth_deriv!(result, matrix, n, device, ::Val{N_rows}) where {N_rows} =
    ClimaComms.@threaded device begin
        for i in @interdependent(axes(matrix, 1)), j in axes(matrix, 2)
            T = eltype(matrix)
            matrix_col =
                ClimaComms.static_shared_memory_array(device, T, N_rows)
            intermediate_result_col =
                ClimaComms.static_shared_memory_array(device, T, N_rows)
            @inbounds begin
                ClimaComms.@sync_interdependent matrix_col[i] = matrix[i, j]
                ClimaComms.@sync_interdependent intermediate_result_col[i] =
                    axis1_nth_deriv(n ÷ 2, matrix_col, i)
                ClimaComms.@sync_interdependent result[i, j] =
                    axis1_nth_deriv(n - n ÷ 2, intermediate_result_col, i)
            end
        end
    end
```

This implementaion has the following performance:
```julia
# TODO: Add performance benchmarks.
```

We can continue to reduce the number of shared memory reads by adding more
intermediate results, though for large values of `n` we will hit another
performance barrier in thread synchronization time. The number of intermediate
results that gives the best performance will depend on the value of `n` and the
characteristics of the GPU.
