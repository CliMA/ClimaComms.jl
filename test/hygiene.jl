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
