"""
    ClimaComms

Abstracts the communications interface for the various CliMA components
to enable use of MPI/SharedArrays/Gasp/etc.

Call `init()` with the type of the communications infrastructure desired.
Create a `RelayBuffer` to hold the data to be communicated; one each for
every neighbor/direction. Call `get_stage()` for each send buffer and fill
them. Call `start()` to initiate communication; `progress()` to drive the
communication progress. To complete communication, call `finish()`. At
this point, call `get_stage()` for each receive buffer and empty them.
"""
module ClimaComms

abstract type AbstractCommsType end
function init end
function start end
function progress end
function finish end

function init(::Type{CT}) where {CT <: AbstractCommsType}
    error("No `init` method defined for $CT")
end

include("relay_buffers.jl")
include("mpi_comms.jl")

end # module
