module ClimaCommsMetalExt

import Metal

import Adapt
import ClimaComms
import ClimaComms: MetalDevice

function ClimaComms.device_synchronize(::MetalDevice)
    Metal.device_synchronize()
    return nothing
end

Base.summary(io::IO, ::MetalDevice) =
    print(io, "$(Metal.device().name) (Metal)")

ClimaComms.device_functional(::MetalDevice) = Metal.functional()

Adapt.adapt_structure(
    ::Type{<:Metal.MtlArray},
    device::ClimaComms.AbstractDevice,
) = MetalDevice()

ClimaComms.array_type(::MetalDevice) = Metal.MtlArray
ClimaComms.free_memory(::MetalDevice) = Metal.free_memory()
ClimaComms.total_memory(::MetalDevice) = Metal.total_memory()
ClimaComms.allowscalar(f, ::MetalDevice, args...; kwargs...) =
    Metal.@allowscalar f(args...; kwargs...)

# Metal methods of the functions that the device-flexible macros call, because
# a macro cannot dispatch on the device type.
ClimaComms.sync(f::F, ::MetalDevice, args...; kwargs...) where {F} =
    Metal.@sync f(args...; kwargs...)
ClimaComms.device_sync(f::F, ::MetalDevice, args...; kwargs...) where {F} =
    Metal.@sync f(args...; kwargs...)
ClimaComms.time(f::F, ::MetalDevice, args...; kwargs...) where {F} =
    Metal.@time f(args...; kwargs...)
# Metal.jl has no `@elapsed` that waits for the GPU.
function ClimaComms.elapsed(f::F, ::MetalDevice, args...; kwargs...) where {F}
    Metal.synchronize()
    return Base.@elapsed Metal.@sync f(args...; kwargs...)
end
ClimaComms.assert(::MetalDevice, cond::C, text::T) where {C, T} =
    cond() ? nothing : throw(AssertionError("ClimaComms.@assert failed"))

# Metal methods of the functions on which the GPU methods of `@threaded` are
# built.
struct ClosureKernel{K, F}
    kernel::K
    f::F
end
call_closure(f) = (f(); nothing)
function ClimaComms.compile_kernel(::MetalDevice, f::F) where {F}
    kernel = Metal.@metal always_inline = true launch = false call_closure(f)
    return ClosureKernel(kernel, f)
end
ClimaComms.launch_kernel(
    ::MetalDevice,
    (; kernel, f)::ClosureKernel,
    blocks,
    threads_in_block,
) = kernel(f; groups = blocks, threads = threads_in_block)
# Metal allows at most typemax(UInt32) threads in each dimension of a grid.
ClimaComms.max_blocks_in_kernel(::MetalDevice, (; kernel)::ClosureKernel) =
    fld(Int(typemax(UInt32)), kernel.maxthreads)
ClimaComms.max_threads_in_block(
    ::MetalDevice,
    (; kernel)::ClosureKernel,
    block_size::Int,
    _,
) = min(block_size, kernel.maxthreads)
# With the default block size, the number of threads in each threadgroup is the
# largest multiple of the SIMD width that is at most the number of threads of
# the loop divided by GROUPS_PER_CORE times the number of GPU cores, from one
# SIMD group to the largest threadgroup of the kernel. Metal.jl returns 0 cores for a device that
# does not report them.
const GROUPS_PER_CORE = 16
function ClimaComms.max_threads_in_block(
    ::MetalDevice,
    (; kernel)::ClosureKernel,
    ::Val{:auto},
    n_threads,
)
    simd_width = kernel.exec_width
    n_groups = GROUPS_PER_CORE * max(Metal.num_gpu_cores(), 1)
    threads = simd_width * max(fld(fld(n_threads, n_groups), simd_width), 1)
    return min(threads, kernel.maxthreads)
end
ClimaComms.thread_index_in_kernel(::MetalDevice) =
    Int(Metal.thread_position_in_grid().x)
ClimaComms.threads_in_kernel(::MetalDevice) = Int(Metal.threads_per_grid().x)

end
