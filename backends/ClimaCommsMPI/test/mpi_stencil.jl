push!(LOAD_PATH, joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using ClimaComms
using ClimaCommsMPI

include(joinpath(@__DIR__, "..", "..", "..", "test", "stencil.jl"))

stencil_test(ClimaCommsMPI.MPICommsContext)
