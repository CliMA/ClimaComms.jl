"""
    MPICommsContext()
    MPICommsContext(device)
    MPICommsContext(device, comm)

A MPI communications context, used for distributed runs.
[`AbstractCPUDevice`](@ref) and [`CUDADevice`](@ref) device options are currently supported.
"""
struct MPICommsContext{D <: AbstractDevice, C} <: AbstractCommsContext
    device::D
    mpicomm::C
end

function MPICommsContext end
