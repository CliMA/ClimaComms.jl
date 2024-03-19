using Artifacts
using LazyArtifacts
using Test

import ClimaComms

const context = ClimaComms.context()

@show expected_path = artifact"socrates"

# Remove the artifact, so that we test that we are downloading it
Base.Filesystem.rm(dirname(expected_path), recursive = true)
@info "Removed artifact"

@testset "Artifact, context: $context" begin
    @test_throws ErrorException @macroexpand ClimaComms.@clima_artifact(
        "socrates"
    )
    @test ClimaComms.@clima_artifact("socrates", context) == expected_path

    # Test with name as a variable
    Base.Filesystem.rm(dirname(expected_path), recursive = true)
    artifact_name = "socrates"

    @test ClimaComms.@clima_artifact(artifact_name, context) == expected_path

    @test_throws ErrorException ClimaComms.@clima_artifact(artifact_name)
end

# Test with a non lazy-artifact

@show expected_path2 = artifact"laskar2004"

# Remove the artifact, so that we test that we are downloading it
Base.Filesystem.rm(dirname(expected_path2), recursive = true)

@testset "Non-lazy artifact, context: $context" begin
    @test ClimaComms.@clima_artifact("laskar2004") == expected_path2

    # Test with name as a variable
    Base.Filesystem.rm(dirname(expected_path2), recursive = true)
    artifact_name = "laskar2004"

    @test ClimaComms.@clima_artifact(artifact_name) == expected_path2
end
