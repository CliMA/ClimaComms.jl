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

"""
    local_communicator(ctx::MPICommsContext)

Internal function to create a new MPI communicator for processes on the same physical node.

The communicator must be freed by `MPI.free`
```
local_comm = ClimaComms.local_communicator(ctx)
# use the communicator
MPI.free(local_comm)
```
"""
function local_communicator end
