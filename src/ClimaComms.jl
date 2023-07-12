"""
    ClimaComms

Abstracts the communications interface for the various CliMA components
in order to:
- support different computational backends (CPUs, GPUs)
- enable the use of different backends as transports (MPI, SharedArrays,
  etc.), and
- transparently support single or double buffering for GPUs, depending
  on whether the transport has the ability to access GPU memory.
"""
module ClimaComms

using Requires
function __init__()
    @static if !isdefined(Base, :get_extension)
        @require BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf" include(
            joinpath("..", "ext", "BenchmarkToolsExt.jl"),
        )
    end
    return nothing
end

include("devices.jl")
include("context.jl")
include("singleton.jl")
include("mpi.jl")

end # module
