var documenterSearchIndex = {"docs":
[{"location":"#ClimaComms","page":"Home","title":"ClimaComms","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = ClimaComms","category":"page"},{"location":"","page":"Home","title":"Home","text":"ClimaComms","category":"page"},{"location":"#ClimaComms.ClimaComms","page":"Home","title":"ClimaComms.ClimaComms","text":"ClimaComms\n\nAbstracts the communications interface for the various CliMA components in order to:\n\nsupport different computational backends (CPUs, GPUs)\nenable the use of different backends as transports (MPI, SharedArrays, etc.), and\ntransparently support single or double buffering for GPUs, depending on whether the transport has the ability to access GPU memory.\n\n\n\n\n\n","category":"module"},{"location":"#Loading","page":"Home","title":"Loading","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ClimaComms.import_required_backends\nClimaComms.cuda_is_required\nClimaComms.mpi_is_required","category":"page"},{"location":"#ClimaComms.cuda_is_required","page":"Home","title":"ClimaComms.cuda_is_required","text":"cuda_is_required()\n\nReturns a Bool indicating if CUDA should be loaded, based on the ENV[\"CLIMACOMMS_DEVICE\"]. See ClimaComms.device for more information.\n\ncuda_is_required() && using CUDA\n\n\n\n\n\n","category":"function"},{"location":"#ClimaComms.mpi_is_required","page":"Home","title":"ClimaComms.mpi_is_required","text":"mpi_is_required()\n\nReturns a Bool indicating if MPI should be loaded, based on the ENV[\"CLIMACOMMS_CONTEXT\"]. See ClimaComms.context for more information.\n\nmpi_is_required() && using MPI\n\n\n\n\n\n","category":"function"},{"location":"#Devices","page":"Home","title":"Devices","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ClimaComms.AbstractDevice\nClimaComms.AbstractCPUDevice\nClimaComms.CPUSingleThreading\nClimaComms.CPUMultiThreading\nClimaComms.CUDADevice\nClimaComms.device\nClimaComms.device_functional\nClimaComms.array_type\nClimaComms.allowscalar\nClimaComms.@threaded\nClimaComms.@time\nClimaComms.@elapsed\nClimaComms.@assert\nClimaComms.@sync\nClimaComms.@cuda_sync","category":"page"},{"location":"#ClimaComms.AbstractDevice","page":"Home","title":"ClimaComms.AbstractDevice","text":"AbstractDevice\n\nThe base type for a device.\n\n\n\n\n\n","category":"type"},{"location":"#ClimaComms.AbstractCPUDevice","page":"Home","title":"ClimaComms.AbstractCPUDevice","text":"AbstractCPUDevice()\n\nAbstract device type for single-threaded and multi-threaded CPU runs.\n\n\n\n\n\n","category":"type"},{"location":"#ClimaComms.CUDADevice","page":"Home","title":"ClimaComms.CUDADevice","text":"CUDADevice()\n\nUse NVIDIA GPU accelarator\n\n\n\n\n\n","category":"type"},{"location":"#ClimaComms.device","page":"Home","title":"ClimaComms.device","text":"ClimaComms.device()\n\nDetermine the device to use depending on the CLIMACOMMS_DEVICE environment variable.\n\nAllowed values:\n\nCPU, single-threaded or multi-threaded depending on the number of threads;\nCPUSingleThreaded,\nCPUMultiThreaded,\nCUDA.\n\nThe default is CPU.\n\n\n\n\n\n","category":"function"},{"location":"#ClimaComms.device_functional","page":"Home","title":"ClimaComms.device_functional","text":"ClimaComms.device_functional(device)\n\nReturn true when the device is correctly set up.\n\n\n\n\n\n","category":"function"},{"location":"#ClimaComms.array_type","page":"Home","title":"ClimaComms.array_type","text":"ClimaComms.array_type(::AbstractDevice)\n\nThe base array type used by the specified device (currently Array or CuArray).\n\n\n\n\n\n","category":"function"},{"location":"#ClimaComms.allowscalar","page":"Home","title":"ClimaComms.allowscalar","text":"allowscalar(f, ::AbstractDevice, args...; kwargs...)\n\nDevice-flexible version of CUDA.@allowscalar.\n\nLowers to\n\nf(args...)\n\nfor CPU devices and\n\nCUDA.@allowscalar f(args...)\n\nfor CUDA devices.\n\nThis is usefully written with closures via\n\nallowscalar(device) do\n    f()\nend\n\n\n\n\n\n","category":"function"},{"location":"#ClimaComms.@threaded","page":"Home","title":"ClimaComms.@threaded","text":"@threaded device for ... end\n\nA threading macro that uses Julia native threading if the device is a CPUMultiThreaded type, otherwise return the original expression without Threads.@threads. This is done to avoid overhead from Threads.@threads, and the device is used (instead of checking Threads.nthreads() == 1) so that this is statically inferred.\n\nReferences\n\nhttps://discourse.julialang.org/t/threads-threads-with-one-thread-how-to-remove-the-overhead/58435\nhttps://discourse.julialang.org/t/overhead-of-threads-threads/53964\n\n\n\n\n\n","category":"macro"},{"location":"#ClimaComms.@time","page":"Home","title":"ClimaComms.@time","text":"@time device expr\n\nDevice-flexible @time.\n\nLowers to\n\n@time expr\n\nfor CPU devices and\n\nCUDA.@time expr\n\nfor CUDA devices.\n\n\n\n\n\n","category":"macro"},{"location":"#ClimaComms.@elapsed","page":"Home","title":"ClimaComms.@elapsed","text":"@elapsed device expr\n\nDevice-flexible @elapsed.\n\nLowers to\n\n@elapsed expr\n\nfor CPU devices and\n\nCUDA.@elapsed expr\n\nfor CUDA devices.\n\n\n\n\n\n","category":"macro"},{"location":"#ClimaComms.@assert","page":"Home","title":"ClimaComms.@assert","text":"@assert device cond [text]\n\nDevice-flexible @assert.\n\nLowers to\n\n@assert cond [text]\n\nfor CPU devices and\n\nCUDA.@cuassert cond [text]\n\nfor CUDA devices.\n\n\n\n\n\n","category":"macro"},{"location":"#ClimaComms.@sync","page":"Home","title":"ClimaComms.@sync","text":"@sync device expr\n\nDevice-flexible @sync.\n\nLowers to\n\n@sync expr\n\nfor CPU devices and\n\nCUDA.@sync expr\n\nfor CUDA devices.\n\nAn example use-case of this might be:\n\nBenchmarkTools.@benchmark begin\n    if ClimaComms.device() isa ClimaComms.CUDADevice\n        CUDA.@sync begin\n            launch_cuda_kernels_or_spawn_tasks!(...)\n        end\n    elseif ClimaComms.device() isa ClimaComms.CPUMultiThreading\n        Base.@sync begin\n            launch_cuda_kernels_or_spawn_tasks!(...)\n        end\n    end\nend\n\nIf the CPU version of the above example does not leverage spawned tasks (which require using Base.sync or Threads.wait to synchronize), then you may want to simply use @cuda_sync.\n\n\n\n\n\n","category":"macro"},{"location":"#ClimaComms.@cuda_sync","page":"Home","title":"ClimaComms.@cuda_sync","text":"@cuda_sync device expr\n\nDevice-flexible CUDA.@sync.\n\nLowers to\n\nexpr\n\nfor CPU devices and\n\nCUDA.@sync expr\n\nfor CUDA devices.\n\n\n\n\n\n","category":"macro"},{"location":"#Contexts","page":"Home","title":"Contexts","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ClimaComms.AbstractCommsContext\nClimaComms.SingletonCommsContext\nClimaComms.MPICommsContext\nClimaComms.AbstractGraphContext\nClimaComms.context\nClimaComms.graph_context","category":"page"},{"location":"#ClimaComms.AbstractCommsContext","page":"Home","title":"ClimaComms.AbstractCommsContext","text":"AbstractCommsContext\n\nThe base type for a communications context. Each backend defines a concrete subtype of this.\n\n\n\n\n\n","category":"type"},{"location":"#ClimaComms.SingletonCommsContext","page":"Home","title":"ClimaComms.SingletonCommsContext","text":"SingletonCommsContext(device=device())\n\nA singleton communications context, used for single-process runs. ClimaComms.CPU and ClimaComms.CUDA device options are currently supported.\n\n\n\n\n\n","category":"type"},{"location":"#ClimaComms.MPICommsContext","page":"Home","title":"ClimaComms.MPICommsContext","text":"MPICommsContext()\nMPICommsContext(device)\nMPICommsContext(device, comm)\n\nA MPI communications context, used for distributed runs. AbstractCPUDevice and CUDADevice device options are currently supported.\n\n\n\n\n\n","category":"type"},{"location":"#ClimaComms.AbstractGraphContext","page":"Home","title":"ClimaComms.AbstractGraphContext","text":"AbstractGraphContext\n\nA context for communicating between processes in a graph.\n\n\n\n\n\n","category":"type"},{"location":"#ClimaComms.context","page":"Home","title":"ClimaComms.context","text":"ClimaComms.context(device=device())\n\nConstruct a default communication context.\n\nBy default, it will try to determine if it is running inside an MPI environment variables are set; if so it will return a MPICommsContext; otherwise it will return a SingletonCommsContext.\n\nBehavior can be overridden by setting the CLIMACOMMS_CONTEXT environment variable to either MPI or SINGLETON.\n\n\n\n\n\n","category":"function"},{"location":"#ClimaComms.graph_context","page":"Home","title":"ClimaComms.graph_context","text":"graph_context(context::AbstractCommsContext,\n              sendarray, sendlengths, sendpids,\n              recvarray, recvlengths, recvpids)\n\nConstruct a communication context for exchanging neighbor data via a graph.\n\nArguments:\n\ncontext: the communication context on which to construct the graph context.\nsendarray: array containing data to send\nsendlengths: list of lengths of data to send to each process ID\nsendpids: list of processor IDs to send\nrecvarray: array to receive data into\nrecvlengths: list of lengths of data to receive from each process ID\nrecvpids: list of processor IDs to receive from\n\nThis should return an AbstractGraphContext object.\n\n\n\n\n\n","category":"function"},{"location":"#Context-operations","page":"Home","title":"Context operations","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ClimaComms.init\nClimaComms.mypid\nClimaComms.iamroot\nClimaComms.nprocs\nClimaComms.abort","category":"page"},{"location":"#ClimaComms.init","page":"Home","title":"ClimaComms.init","text":"(pid, nprocs) = init(ctx::AbstractCommsContext)\n\nPerform any necessary initialization for the specified backend. Return a tuple of the processor ID and the number of participating processors.\n\n\n\n\n\n","category":"function"},{"location":"#ClimaComms.mypid","page":"Home","title":"ClimaComms.mypid","text":"mypid(ctx::AbstractCommsContext)\n\nReturn the processor ID.\n\n\n\n\n\n","category":"function"},{"location":"#ClimaComms.iamroot","page":"Home","title":"ClimaComms.iamroot","text":"iamroot(ctx::AbstractCommsContext)\n\nReturn true if the calling processor is the root processor.\n\n\n\n\n\n","category":"function"},{"location":"#ClimaComms.nprocs","page":"Home","title":"ClimaComms.nprocs","text":"nprocs(ctx::AbstractCommsContext)\n\nReturn the number of participating processors.\n\n\n\n\n\n","category":"function"},{"location":"#ClimaComms.abort","page":"Home","title":"ClimaComms.abort","text":"abort(ctx::CC, status::Int) where {CC <: AbstractCommsContext}\n\nTerminate the caller and all participating processors with the specified status.\n\n\n\n\n\n","category":"function"},{"location":"#Collective-operations","page":"Home","title":"Collective operations","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ClimaComms.barrier\nClimaComms.reduce\nClimaComms.reduce!\nClimaComms.allreduce\nClimaComms.allreduce!\nClimaComms.bcast","category":"page"},{"location":"#ClimaComms.barrier","page":"Home","title":"ClimaComms.barrier","text":"barrier(ctx::CC) where {CC <: AbstractCommsContext}\n\nPerform a global synchronization across all participating processors.\n\n\n\n\n\n","category":"function"},{"location":"#ClimaComms.reduce","page":"Home","title":"ClimaComms.reduce","text":"reduce(ctx::CC, val, op) where {CC <: AbstractCommsContext}\n\nPerform a reduction across all participating processors, using op as the reduction operator and val as this rank's reduction value. Return the result to the first processor only.\n\n\n\n\n\n","category":"function"},{"location":"#ClimaComms.reduce!","page":"Home","title":"ClimaComms.reduce!","text":"reduce!(ctx::CC, sendbuf, recvbuf, op)\nreduce!(ctx::CC, sendrecvbuf, op)\n\nPerforms elementwise reduction using the operator op on the buffer sendbuf, storing the result in the recvbuf of the process. If only one sendrecvbuf buffer is provided, then the operation is performed in-place.\n\n\n\n\n\n","category":"function"},{"location":"#ClimaComms.allreduce","page":"Home","title":"ClimaComms.allreduce","text":"allreduce(ctx::CC, sendbuf, op)\n\nPerforms elementwise reduction using the operator op on the buffer sendbuf, allocating a new array for the result. sendbuf can also be a scalar, in which case recvbuf will be a value of the same type.\n\n\n\n\n\n","category":"function"},{"location":"#ClimaComms.allreduce!","page":"Home","title":"ClimaComms.allreduce!","text":"allreduce!(ctx::CC, sendbuf, recvbuf, op)\nallreduce!(ctx::CC, sendrecvbuf, op)\n\nPerforms elementwise reduction using the operator op on the buffer sendbuf, storing the result in the recvbuf of all processes in the group. Allreduce! is equivalent to a Reduce! operation followed by a Bcast!, but can lead to better performance. If only one sendrecvbuf buffer is provided, then the operation is performed in-place.\n\n\n\n\n\n","category":"function"},{"location":"#ClimaComms.bcast","page":"Home","title":"ClimaComms.bcast","text":"bcast(ctx::AbstractCommsContext, object)\n\nBroadcast object from the root process to all other processes. The value of object on non-root processes is ignored.\n\n\n\n\n\n","category":"function"},{"location":"#Graph-exchange","page":"Home","title":"Graph exchange","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ClimaComms.start\nClimaComms.progress\nClimaComms.finish","category":"page"},{"location":"#ClimaComms.start","page":"Home","title":"ClimaComms.start","text":"start(ctx::AbstractGraphContext)\n\nInitiate graph data exchange.\n\n\n\n\n\n","category":"function"},{"location":"#ClimaComms.progress","page":"Home","title":"ClimaComms.progress","text":"progress(ctx::AbstractGraphContext)\n\nDrive communication. Call after start to ensure that communication proceeds asynchronously.\n\n\n\n\n\n","category":"function"},{"location":"#ClimaComms.finish","page":"Home","title":"ClimaComms.finish","text":"finish(ctx::AbstractGraphContext)\n\nComplete the communications step begun by start(). After this returns, data received from all neighbors will be available in the stage areas of each neighbor's receive buffer.\n\n\n\n\n\n","category":"function"}]
}
