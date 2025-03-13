#!/usr/bin/env julia

# Set environment variables for CUDA device testing
ENV["CLIMACOMMS_DEVICE"] = "CUDA"
ENV["CLIMACOMMS_TEST_DEVICE"] = "CUDA"

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Test
@info "Running ClimaComms.jl tests with CUDA device"
include("runtests.jl")
