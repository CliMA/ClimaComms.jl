#!/usr/bin/env julia

# Set environment variables for Metal device testing
ENV["CLIMACOMMS_DEVICE"] = "Metal"
ENV["CLIMACOMMS_TEST_DEVICE"] = "Metal"

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Test
@info "Running ClimaComms.jl tests with Metal device"
include("runtests.jl")
