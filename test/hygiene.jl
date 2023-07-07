using BenchmarkTools # load src/weak_deps/benchmark_tools.jl
import ClimaComms as CC
dev = CC.device()
CC.@threaded dev for i in 1:2
    1
end

CC.@time dev for i in 1:2
    sin.(rand(10))
end

CC.@elapsed dev for i in 1:2
    sin.(rand(10))
end

CC.@benchmark dev for i in 1:2
    sin.(rand(10))
end
