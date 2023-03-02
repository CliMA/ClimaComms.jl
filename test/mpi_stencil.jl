using ClimaComms, Test

@test ClimaComms.context() isa ClimaComms.MPICommsContext

include("stencil.jl")

stencil_test(ClimaComms.MPICommsContext())
stencil_test(ClimaComms.MPICommsContext(), persistent = true)
