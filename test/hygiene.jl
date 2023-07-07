import ClimaComms as CC
dev = CC.device()
CC.@threaded dev for i in 1:2
    1
end
