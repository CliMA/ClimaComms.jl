"""
    MPICommsContext()
    MPICommsContext(device)
    MPICommsContext(device, comm)

An MPI communications context, used for distributed runs.
[`AbstractCPUDevice`](@ref) and [`CUDADevice`](@ref) device options are
currently supported. The `comm` argument defaults to `MPI.COMM_WORLD`.

`MPI.jl` must be loaded for this context to be usable; see
[`@import_required_backends`](@ref).

# Fields
- `device`: the [`AbstractDevice`](@ref) on which computations run.
- `mpicomm`: the MPI communicator (an `MPI.Comm`).
"""
struct MPICommsContext{D <: AbstractDevice, C} <: AbstractCommsContext
    device::D
    mpicomm::C
end

function MPICommsContext end

"""
    ClimaComms.local_communicator(ctx::MPICommsContext)
    ClimaComms.local_communicator(f, ctx::MPICommsContext)

Create a new MPI communicator containing the processes on the same
physical node as the caller. In the single-argument form, the caller is
responsible for freeing the communicator; the two-argument (do-block) form
calls `f` on the communicator and then frees it.

Called from [`init`](@ref) to assign GPUs to the MPI ranks on each node.

# Examples
```julia
ClimaComms.local_communicator(ctx) do local_comm
    ClimaComms._assign_device(
        ClimaComms.device(ctx),
        MPI.Comm_rank(local_comm),
    )
end
```
"""
function local_communicator end
