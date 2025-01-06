"""
    MPICommsContext()
    MPICommsContext(device)
    MPICommsContext(device, comm)

A MPI communications context, used for distributed runs.
[`AbstractCPUDevice`](@ref) and [`CUDADevice`](@ref) device options are currently supported.
"""
struct MPICommsContext{D <: AbstractDevice} <: AbstractCommsContext
    device::D
    function MPICommsContext(dev::AbstractDevice = device())
        set_mpicomm!()
        return new{typeof(dev)}(dev)
    end
end

function MPICommsContext end

function set_mpicomm! end
function mpicomm end
