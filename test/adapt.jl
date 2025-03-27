import ClimaComms
import Adapt: adapt

@testset "Adapt" begin
    # Test with CPU device
    cpu_device = ClimaComms.CPUSingleThreaded()

    # Test with simple arrays
    x_array = [1, 2, 3, 4]

    # Test that adapt(device, x) correctly forwards to adapt(array_type(device), x)
    @test adapt(cpu_device, x_array) ==
          adapt(ClimaComms.array_type(cpu_device), x_array)
    @test adapt(cpu_device, x_array) == adapt(Array, x_array)
    @test adapt(ClimaComms.SingletonCommsContext(cpu_device), x_array) ==
          adapt(cpu_device, x_array)

    # Test with nested structures
    nested_data = (a = [1, 2, 3], b = Dict(:x => [4, 5, 6]))
    @test adapt(cpu_device, nested_data) == adapt(Array, nested_data)

    # Test with scalar values (should remain unchanged)
    @test adapt(cpu_device, 42) == 42
    @test adapt(cpu_device, "test") == "test"
end
