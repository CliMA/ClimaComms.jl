import ClimaComms as CC
function test_macro_hyhiene(dev)
    n = 3 # tests that we can reach variables defined in scope
    CC.@threaded dev for i in 1:n
        if Threads.nthreads() > 1
            @show Threads.threadid()
        end
    end

    CC.@time dev for i in 1:n
        sin.(rand(10))
    end

    CC.@elapsed dev for i in 1:n
        sin.(rand(10))
    end
end
dev = CC.device()

test_macro_hyhiene(dev)
