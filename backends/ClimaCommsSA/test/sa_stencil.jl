push!(LOAD_PATH, joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using ClimaComms
using ClimaCommsSA

include(joinpath(@__DIR__, "..", "..", "..", "test", "stencil.jl"))

stencil_test(ClimaCommsSA.SACommsContext)
