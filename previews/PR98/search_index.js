var documenterSearchIndex = {"docs":
[{"location":"apis/#APIs","page":"APIs","title":"APIs","text":"","category":"section"},{"location":"apis/","page":"APIs","title":"APIs","text":"CurrentModule = ClimaComms","category":"page"},{"location":"apis/","page":"APIs","title":"APIs","text":"ClimaComms","category":"page"},{"location":"apis/#ClimaComms.ClimaComms","page":"APIs","title":"ClimaComms.ClimaComms","text":"ClimaComms\n\nAbstracts the communications interface for the various CliMA components in order to:\n\nsupport different computational backends (CPUs, GPUs)\nenable the use of different backends as transports (MPI, SharedArrays, etc.), and\ntransparently support single or double buffering for GPUs, depending on whether the transport has the ability to access GPU memory.\n\n\n\n\n\n","category":"module"},{"location":"apis/#Loading","page":"APIs","title":"Loading","text":"","category":"section"},{"location":"apis/","page":"APIs","title":"APIs","text":"ClimaComms.@import_required_backends\nClimaComms.cuda_is_required\nClimaComms.mpi_is_required","category":"page"},{"location":"apis/#ClimaComms.@import_required_backends","page":"APIs","title":"ClimaComms.@import_required_backends","text":"ClimaComms.@import_required_backends\n\nIf the desired context is MPI (as determined by ClimaComms.context()), try loading MPI.jl. If the desired device is CUDA (as determined by ClimaComms.device()), try loading CUDA.jl.\n\n\n\n\n\n","category":"macro"},{"location":"apis/#ClimaComms.cuda_is_required","page":"APIs","title":"ClimaComms.cuda_is_required","text":"cuda_is_required()\n\nReturns a Bool indicating if CUDA should be loaded, based on the ENV[\"CLIMACOMMS_DEVICE\"]. See ClimaComms.device for more information.\n\ncuda_is_required() && using CUDA\n\n\n\n\n\n","category":"function"},{"location":"apis/#ClimaComms.mpi_is_required","page":"APIs","title":"ClimaComms.mpi_is_required","text":"mpi_is_required()\n\nReturns a Bool indicating if MPI should be loaded, based on the ENV[\"CLIMACOMMS_CONTEXT\"]. See ClimaComms.context for more information.\n\nmpi_is_required() && using MPI\n\n\n\n\n\n","category":"function"},{"location":"apis/#Devices","page":"APIs","title":"Devices","text":"","category":"section"},{"location":"apis/","page":"APIs","title":"APIs","text":"ClimaComms.AbstractDevice\nClimaComms.AbstractCPUDevice\nClimaComms.CPUSingleThreaded\nClimaComms.CPUMultiThreaded\nClimaComms.CUDADevice\nClimaComms.device\nClimaComms.device_functional\nClimaComms.array_type\nClimaComms.allowscalar\nClimaComms.@threaded\nClimaComms.@time\nClimaComms.@elapsed\nClimaComms.@assert\nClimaComms.@sync\nClimaComms.@cuda_sync","category":"page"},{"location":"apis/#ClimaComms.AbstractDevice","page":"APIs","title":"ClimaComms.AbstractDevice","text":"AbstractDevice\n\nThe base type for a device.\n\n\n\n\n\n","category":"type"},{"location":"apis/#ClimaComms.AbstractCPUDevice","page":"APIs","title":"ClimaComms.AbstractCPUDevice","text":"AbstractCPUDevice()\n\nAbstract device type for single-threaded and multi-threaded CPU runs.\n\n\n\n\n\n","category":"type"},{"location":"apis/#ClimaComms.CPUSingleThreaded","page":"APIs","title":"ClimaComms.CPUSingleThreaded","text":"CPUSingleThreaded()\n\nUse the CPU with single thread.\n\n\n\n\n\n","category":"type"},{"location":"apis/#ClimaComms.CPUMultiThreaded","page":"APIs","title":"ClimaComms.CPUMultiThreaded","text":"CPUMultiThreaded()\n\nUse the CPU with multiple thread.\n\n\n\n\n\n","category":"type"},{"location":"apis/#ClimaComms.CUDADevice","page":"APIs","title":"ClimaComms.CUDADevice","text":"CUDADevice()\n\nUse NVIDIA GPU accelarator\n\n\n\n\n\n","category":"type"},{"location":"apis/#ClimaComms.device","page":"APIs","title":"ClimaComms.device","text":"ClimaComms.device()\n\nDetermine the device to use depending on the CLIMACOMMS_DEVICE environment variable.\n\nAllowed values:\n\nCPU, single-threaded or multi-threaded depending on the number of threads;\nCPUSingleThreaded,\nCPUMultiThreaded,\nCUDA.\n\nThe default is CPU.\n\n\n\n\n\n","category":"function"},{"location":"apis/#ClimaComms.device_functional","page":"APIs","title":"ClimaComms.device_functional","text":"ClimaComms.device_functional(device)\n\nReturn true when the device is correctly set up.\n\n\n\n\n\n","category":"function"},{"location":"apis/#ClimaComms.array_type","page":"APIs","title":"ClimaComms.array_type","text":"ClimaComms.array_type(::AbstractDevice)\n\nThe base array type used by the specified device (currently Array or CuArray).\n\n\n\n\n\n","category":"function"},{"location":"apis/#ClimaComms.allowscalar","page":"APIs","title":"ClimaComms.allowscalar","text":"allowscalar(f, ::AbstractDevice, args...; kwargs...)\n\nDevice-flexible version of CUDA.@allowscalar.\n\nLowers to\n\nf(args...)\n\nfor CPU devices and\n\nCUDA.@allowscalar f(args...)\n\nfor CUDA devices.\n\nThis is usefully written with closures via\n\nallowscalar(device) do\n    f()\nend\n\n\n\n\n\n","category":"function"},{"location":"apis/#ClimaComms.@threaded","page":"APIs","title":"ClimaComms.@threaded","text":"@threaded device for ... end\n\nA threading macro that uses Julia native threading if the device is a CPUMultiThreaded type, otherwise return the original expression without Threads.@threads. This is done to avoid overhead from Threads.@threads, and the device is used (instead of checking Threads.nthreads() == 1) so that this is statically inferred.\n\nReferences\n\nhttps://discourse.julialang.org/t/threads-threads-with-one-thread-how-to-remove-the-overhead/58435\nhttps://discourse.julialang.org/t/overhead-of-threads-threads/53964\n\n\n\n\n\n","category":"macro"},{"location":"apis/#ClimaComms.@time","page":"APIs","title":"ClimaComms.@time","text":"@time device expr\n\nDevice-flexible @time.\n\nLowers to\n\n@time expr\n\nfor CPU devices and\n\nCUDA.@time expr\n\nfor CUDA devices.\n\n\n\n\n\n","category":"macro"},{"location":"apis/#ClimaComms.@elapsed","page":"APIs","title":"ClimaComms.@elapsed","text":"@elapsed device expr\n\nDevice-flexible @elapsed.\n\nLowers to\n\n@elapsed expr\n\nfor CPU devices and\n\nCUDA.@elapsed expr\n\nfor CUDA devices.\n\n\n\n\n\n","category":"macro"},{"location":"apis/#ClimaComms.@assert","page":"APIs","title":"ClimaComms.@assert","text":"@assert device cond [text]\n\nDevice-flexible @assert.\n\nLowers to\n\n@assert cond [text]\n\nfor CPU devices and\n\nCUDA.@cuassert cond [text]\n\nfor CUDA devices.\n\n\n\n\n\n","category":"macro"},{"location":"apis/#ClimaComms.@sync","page":"APIs","title":"ClimaComms.@sync","text":"@sync device expr\n\nDevice-flexible @sync.\n\nLowers to\n\n@sync expr\n\nfor CPU devices and\n\nCUDA.@sync expr\n\nfor CUDA devices.\n\nAn example use-case of this might be:\n\nBenchmarkTools.@benchmark begin\n    if ClimaComms.device() isa ClimaComms.CUDADevice\n        CUDA.@sync begin\n            launch_cuda_kernels_or_spawn_tasks!(...)\n        end\n    elseif ClimaComms.device() isa ClimaComms.CPUMultiThreading\n        Base.@sync begin\n            launch_cuda_kernels_or_spawn_tasks!(...)\n        end\n    end\nend\n\nIf the CPU version of the above example does not leverage spawned tasks (which require using Base.sync or Threads.wait to synchronize), then you may want to simply use @cuda_sync.\n\n\n\n\n\n","category":"macro"},{"location":"apis/#ClimaComms.@cuda_sync","page":"APIs","title":"ClimaComms.@cuda_sync","text":"@cuda_sync device expr\n\nDevice-flexible CUDA.@sync.\n\nLowers to\n\nexpr\n\nfor CPU devices and\n\nCUDA.@sync expr\n\nfor CUDA devices.\n\n\n\n\n\n","category":"macro"},{"location":"apis/#Contexts","page":"APIs","title":"Contexts","text":"","category":"section"},{"location":"apis/","page":"APIs","title":"APIs","text":"ClimaComms.AbstractCommsContext\nClimaComms.SingletonCommsContext\nClimaComms.MPICommsContext\nClimaComms.AbstractGraphContext\nClimaComms.context\nClimaComms.graph_context","category":"page"},{"location":"apis/#ClimaComms.AbstractCommsContext","page":"APIs","title":"ClimaComms.AbstractCommsContext","text":"AbstractCommsContext\n\nThe base type for a communications context. Each backend defines a concrete subtype of this.\n\n\n\n\n\n","category":"type"},{"location":"apis/#ClimaComms.SingletonCommsContext","page":"APIs","title":"ClimaComms.SingletonCommsContext","text":"SingletonCommsContext(device=device())\n\nA singleton communications context, used for single-process runs. ClimaComms.AbstractCPUDevice and ClimaComms.CUDADevice device options are currently supported.\n\n\n\n\n\n","category":"type"},{"location":"apis/#ClimaComms.MPICommsContext","page":"APIs","title":"ClimaComms.MPICommsContext","text":"MPICommsContext()\nMPICommsContext(device)\nMPICommsContext(device, comm)\n\nA MPI communications context, used for distributed runs. AbstractCPUDevice and CUDADevice device options are currently supported.\n\n\n\n\n\n","category":"type"},{"location":"apis/#ClimaComms.AbstractGraphContext","page":"APIs","title":"ClimaComms.AbstractGraphContext","text":"AbstractGraphContext\n\nA context for communicating between processes in a graph.\n\n\n\n\n\n","category":"type"},{"location":"apis/#ClimaComms.context","page":"APIs","title":"ClimaComms.context","text":"ClimaComms.context(device=device())\n\nConstruct a default communication context.\n\nBy default, it will try to determine if it is running inside an MPI environment variables are set; if so it will return a MPICommsContext; otherwise it will return a SingletonCommsContext.\n\nBehavior can be overridden by setting the CLIMACOMMS_CONTEXT environment variable to either MPI or SINGLETON.\n\n\n\n\n\n","category":"function"},{"location":"apis/#ClimaComms.graph_context","page":"APIs","title":"ClimaComms.graph_context","text":"graph_context(context::AbstractCommsContext,\n              sendarray, sendlengths, sendpids,\n              recvarray, recvlengths, recvpids)\n\nConstruct a communication context for exchanging neighbor data via a graph.\n\nArguments:\n\ncontext: the communication context on which to construct the graph context.\nsendarray: array containing data to send\nsendlengths: list of lengths of data to send to each process ID\nsendpids: list of processor IDs to send\nrecvarray: array to receive data into\nrecvlengths: list of lengths of data to receive from each process ID\nrecvpids: list of processor IDs to receive from\n\nThis should return an AbstractGraphContext object.\n\n\n\n\n\n","category":"function"},{"location":"apis/#Context-operations","page":"APIs","title":"Context operations","text":"","category":"section"},{"location":"apis/","page":"APIs","title":"APIs","text":"ClimaComms.init\nClimaComms.mypid\nClimaComms.iamroot\nClimaComms.nprocs\nClimaComms.abort","category":"page"},{"location":"apis/#ClimaComms.init","page":"APIs","title":"ClimaComms.init","text":"(pid, nprocs) = init(ctx::AbstractCommsContext)\n\nPerform any necessary initialization for the specified backend. Return a tuple of the processor ID and the number of participating processors.\n\n\n\n\n\n","category":"function"},{"location":"apis/#ClimaComms.mypid","page":"APIs","title":"ClimaComms.mypid","text":"mypid(ctx::AbstractCommsContext)\n\nReturn the processor ID.\n\n\n\n\n\n","category":"function"},{"location":"apis/#ClimaComms.iamroot","page":"APIs","title":"ClimaComms.iamroot","text":"iamroot(ctx::AbstractCommsContext)\n\nReturn true if the calling processor is the root processor.\n\n\n\n\n\n","category":"function"},{"location":"apis/#ClimaComms.nprocs","page":"APIs","title":"ClimaComms.nprocs","text":"nprocs(ctx::AbstractCommsContext)\n\nReturn the number of participating processors.\n\n\n\n\n\n","category":"function"},{"location":"apis/#ClimaComms.abort","page":"APIs","title":"ClimaComms.abort","text":"abort(ctx::CC, status::Int) where {CC <: AbstractCommsContext}\n\nTerminate the caller and all participating processors with the specified status.\n\n\n\n\n\n","category":"function"},{"location":"apis/#Collective-operations","page":"APIs","title":"Collective operations","text":"","category":"section"},{"location":"apis/","page":"APIs","title":"APIs","text":"ClimaComms.barrier\nClimaComms.reduce\nClimaComms.reduce!\nClimaComms.allreduce\nClimaComms.allreduce!\nClimaComms.bcast","category":"page"},{"location":"apis/#ClimaComms.barrier","page":"APIs","title":"ClimaComms.barrier","text":"barrier(ctx::CC) where {CC <: AbstractCommsContext}\n\nPerform a global synchronization across all participating processors.\n\n\n\n\n\n","category":"function"},{"location":"apis/#ClimaComms.reduce","page":"APIs","title":"ClimaComms.reduce","text":"reduce(ctx::CC, val, op) where {CC <: AbstractCommsContext}\n\nPerform a reduction across all participating processors, using op as the reduction operator and val as this rank's reduction value. Return the result to the first processor only.\n\n\n\n\n\n","category":"function"},{"location":"apis/#ClimaComms.reduce!","page":"APIs","title":"ClimaComms.reduce!","text":"reduce!(ctx::CC, sendbuf, recvbuf, op)\nreduce!(ctx::CC, sendrecvbuf, op)\n\nPerforms elementwise reduction using the operator op on the buffer sendbuf, storing the result in the recvbuf of the process. If only one sendrecvbuf buffer is provided, then the operation is performed in-place.\n\n\n\n\n\n","category":"function"},{"location":"apis/#ClimaComms.allreduce","page":"APIs","title":"ClimaComms.allreduce","text":"allreduce(ctx::CC, sendbuf, op)\n\nPerforms elementwise reduction using the operator op on the buffer sendbuf, allocating a new array for the result. sendbuf can also be a scalar, in which case recvbuf will be a value of the same type.\n\n\n\n\n\n","category":"function"},{"location":"apis/#ClimaComms.allreduce!","page":"APIs","title":"ClimaComms.allreduce!","text":"allreduce!(ctx::CC, sendbuf, recvbuf, op)\nallreduce!(ctx::CC, sendrecvbuf, op)\n\nPerforms elementwise reduction using the operator op on the buffer sendbuf, storing the result in the recvbuf of all processes in the group. Allreduce! is equivalent to a Reduce! operation followed by a Bcast!, but can lead to better performance. If only one sendrecvbuf buffer is provided, then the operation is performed in-place.\n\n\n\n\n\n","category":"function"},{"location":"apis/#ClimaComms.bcast","page":"APIs","title":"ClimaComms.bcast","text":"bcast(ctx::AbstractCommsContext, object)\n\nBroadcast object from the root process to all other processes. The value of object on non-root processes is ignored.\n\n\n\n\n\n","category":"function"},{"location":"apis/#Graph-exchange","page":"APIs","title":"Graph exchange","text":"","category":"section"},{"location":"apis/","page":"APIs","title":"APIs","text":"ClimaComms.start\nClimaComms.progress\nClimaComms.finish","category":"page"},{"location":"apis/#ClimaComms.start","page":"APIs","title":"ClimaComms.start","text":"start(ctx::AbstractGraphContext)\n\nInitiate graph data exchange.\n\n\n\n\n\n","category":"function"},{"location":"apis/#ClimaComms.progress","page":"APIs","title":"ClimaComms.progress","text":"progress(ctx::AbstractGraphContext)\n\nDrive communication. Call after start to ensure that communication proceeds asynchronously.\n\n\n\n\n\n","category":"function"},{"location":"apis/#ClimaComms.finish","page":"APIs","title":"ClimaComms.finish","text":"finish(ctx::AbstractGraphContext)\n\nComplete the communications step begun by start(). After this returns, data received from all neighbors will be available in the stage areas of each neighbor's receive buffer.\n\n\n\n\n\n","category":"function"},{"location":"faqs/#Frequently-Asked-Questions","page":"Frequently Asked Questions","title":"Frequently Asked Questions","text":"","category":"section"},{"location":"faqs/#How-do-I-run-my-simulation-on-a-GPU?","page":"Frequently Asked Questions","title":"How do I run my simulation on a GPU?","text":"","category":"section"},{"location":"faqs/","page":"Frequently Asked Questions","title":"Frequently Asked Questions","text":"Set the environment variable CLIMACOMMS_DEVICE to CUDA. This can be accomplished in your Julia script with (at the top)","category":"page"},{"location":"faqs/","page":"Frequently Asked Questions","title":"Frequently Asked Questions","text":"ENV[\"CLIMACOMMS_DEVICE\"] = \"CUDA\"","category":"page"},{"location":"faqs/","page":"Frequently Asked Questions","title":"Frequently Asked Questions","text":"or calling","category":"page"},{"location":"faqs/","page":"Frequently Asked Questions","title":"Frequently Asked Questions","text":"export CLIMACOMMS_DEVICE=\"CUDA\"","category":"page"},{"location":"faqs/","page":"Frequently Asked Questions","title":"Frequently Asked Questions","text":"in your shell (outside of Julia, no spaces).","category":"page"},{"location":"faqs/#My-simulation-does-not-start-and-crashes-with-a-MPI-error.-I-don't-want-to-run-with-MPI.-What-should-I-do?","page":"Frequently Asked Questions","title":"My simulation does not start and crashes with a MPI error. I don't want to run with MPI. What should I do?","text":"","category":"section"},{"location":"faqs/","page":"Frequently Asked Questions","title":"Frequently Asked Questions","text":"ClimaComms tries to be smart and select the best configuration for your run. Sometimes, it fails. In this case, you can force ClimaComms to ignore MPI with","category":"page"},{"location":"faqs/","page":"Frequently Asked Questions","title":"Frequently Asked Questions","text":"ENV[\"CLIMACOMMS_CONTEXT\"] = \"SINGLETON\"","category":"page"},{"location":"faqs/","page":"Frequently Asked Questions","title":"Frequently Asked Questions","text":"at the top of your Julia script or by calling","category":"page"},{"location":"faqs/","page":"Frequently Asked Questions","title":"Frequently Asked Questions","text":"export CLIMACOMMS_CONTEXT=\"SINGLETON\"","category":"page"},{"location":"faqs/","page":"Frequently Asked Questions","title":"Frequently Asked Questions","text":"in your shell (outside of Julia, no spaces).","category":"page"},{"location":"faqs/#My-code-is-saying-something-about-ClimaComms.@import_required_backends,-what-does-that-mean?","page":"Frequently Asked Questions","title":"My code is saying something about ClimaComms.@import_required_backends, what does that mean?","text":"","category":"section"},{"location":"faqs/","page":"Frequently Asked Questions","title":"Frequently Asked Questions","text":"When you are using the environment variables to control the execution of your script, ClimaComms can detect that some important packages are not loaded. For example, ClimaComms will emit an error if you set CLIMACOMMS_DEVICE=\"CUDA\" but do not import CUDA.jl in your code.","category":"page"},{"location":"faqs/","page":"Frequently Asked Questions","title":"Frequently Asked Questions","text":"ClimaComms provides a macro, ClimaComms.@import_required_backends, that you can add at the top of your scripts to automatically load the required packages when needed. Note, the packages have to be in your Julia environment, so you might install packages like MPI.jl and CUDA.jl.","category":"page"},{"location":"#ClimaComms","page":"Home","title":"ClimaComms","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ClimaComms.jl is a small package that provides abstractions for different computing devices and environments. ClimaComms.jl is use extensively by CliMA packages to control where and how simulations are run (e.g., one on core, on multiple GPUs, et cetera).","category":"page"},{"location":"","page":"Home","title":"Home","text":"This page highlights the most important user-facing ClimaComms concepts. If you are using ClimaComms as a developer, refer to the Developing with ClimaComms page. For a detailed list of all the functions and objects implemented, the APIs page collects all of them.","category":"page"},{"location":"#Devices-and-Contexts","page":"Home","title":"Devices and Contexts","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The two most important objects in ClimaComms.jl are the Device and the Context.","category":"page"},{"location":"","page":"Home","title":"Home","text":"A Device identifies a computing device, a piece of hardware that will be executing some code. The Devices currently implemented are","category":"page"},{"location":"","page":"Home","title":"Home","text":"CPUSingleThreaded, for a CPU core with a single thread;\nCUDADevice, for a single CUDA-enabled GPU.","category":"page"},{"location":"","page":"Home","title":"Home","text":"warn: Warn\nCPUMultiThreaded is also available, but this device is not actively used or developed.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Devices are part of Contexts, objects that contain information require for multiple Devices to communicate. Implemented Contexts are","category":"page"},{"location":"","page":"Home","title":"Home","text":"SingletonCommsContext, when there is no parallelism;\nMPICommsContext , for a MPI-parallelized runs.","category":"page"},{"location":"","page":"Home","title":"Home","text":"To choose a device and a context, most CliMA packages use the device() and context() functions. These functions look at specific environment variables and set the device and context accordingly. By default, the CPUSingleThreaded device is chosen and the context is set to SingletonCommsContext unless ClimaComms detects being run in a standard MPI launcher (as srun or mpiexec).","category":"page"},{"location":"","page":"Home","title":"Home","text":"For example, to run a simulation on a GPU, run julia as","category":"page"},{"location":"","page":"Home","title":"Home","text":"export CLIMACOMMS_DEVICE=\"CUDA\"\nexport CLIMACOMMS_CONTEXT=\"SINGLETON\"\n# call/open julia as usual","category":"page"},{"location":"","page":"Home","title":"Home","text":"note: Note\nThere might be other ways to control the device and context. Please, refer to the documentation of the specific package to learn more.","category":"page"},{"location":"#Running-with-MPI/CUDA","page":"Home","title":"Running with MPI/CUDA","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"CliMA packages do not depend directly on MPI or CUDA, so, if you want to run your simulation in parallel mode and/or on GPUs, you will need to install some packages separately.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For parallel simulations, MPI.jl, and for GPU runs, CUDA.jl. You can install these packages in your base environment","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia -E \"using Pkg; Pkg.add(\\\"CUDA\\\"); Pkg.add(\\\"MPI\\\")\"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Some packages come with environments that includes all possible backends (typically .buildkite). You can also consider directly using those environments.","category":"page"},{"location":"internals/#Developing-with-ClimaComms","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"","category":"section"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"This page goes into more depth about ClimaComms in CliMA packages.","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"First, we will describe what Devices and Contexts are.","category":"page"},{"location":"internals/#Devices","page":"Developing with ClimaComms","title":"Devices","text":"","category":"section"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"Devices identify a specific type of computing hardware (e.g., a CPU/a NVidia GPU, et cetera). The Devices implemented are","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"CPUSingleThreaded, for a CPU core with a single thread;\nCUDADevice, for a single CUDA GPU.","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"Devices in ClimaComms are singletons, meaning that they do not contain any data: they are used exclusively to determine what implementation of a given function or data structure should be used. For example, let us implement a function to allocate an array what on the CPU or on the GPU depending on the Device:","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"import ClimaComms: CPUSingleThreaded, CUDADevice\nimport CUDA\n\nfunction allocate(device::CPUSingleThreaded, what)\n    return Array(what)\nend\n\nfunction allocate(device::CUDADevice, what)\n    return CUDA.CuArray(what)\nend","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"If we want to allocate memory on the GPU, we would do something like:","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"julia> allocate(CUDADevice(), [1, 2, 3])\nCUDA.CuArray([1, 2, 3])","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"Low-level CliMA code often needs to implement different methods for different Devices (e.g., in ClimaCore), but this level of specialization is often not required at higher levels.","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"Higher-level code often interacts with Devices through ClimaComms functions such time or sync. These functions implement device-agnostic operations. For instance, the proper way to compute how long a given expression takes to compute is","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"import ClimaComms: @time\n\ndevice = ClimaComms.device()  # or the device of interest\n\n@time device my_expr","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"This will ensure that the correct time is computed (e.g., the time to run a GPU kernel, and not the time to launch the kernel).","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"For a complete list of such functions, consult the APIs page.","category":"page"},{"location":"internals/#Contexts","page":"Developing with ClimaComms","title":"Contexts","text":"","category":"section"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"A Context contains the information needed for multiple devices to communicate. For simulations with only one device, SingletonCommsContext simply contains an instance of an AbstractDevice. For MPI simulations, the context contains the MPI communicator as well.","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"Contexts specify devices and form of parallelism, so they are often passed around in both low-level and higher-level code.","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"ClimaComms provide functions that are context-agnostic. For instance, reduce applies a given function to an array across difference processes and collects the result. Let us see an example","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"import ClimaComms\nClimaComms.@import_required_backends\n\ncontext = ClimaComms.context()  # Default context (read from environment variables)\ndevice = ClimaComms.device(context)  # Default device\n\nmypid, nprocs = ClimaComms.init(context)\n\nArrayType = ClimaComms.array_type(device)\n\nmy_array = mypid * ArrayType([1, 1, 1])\n\nreduced_array = ClimaComms.reduce(context, my_array, +)\nClimaComms.iamroot(context) && @show reduced_array","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"@import_required_backends is responsible for loading relevant libraries, for more information refer to the Backends and extensions section.","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"In this snippet, we obtained the default context from environment variables using the context function. As developers, we do not know whether this code is being run on a single process or multiple, so took the more generic stance that the code might be run on several processes. When several processes are being used, the same code is being run by parallel Julia instances, each with a different process id (pid). The line julia mypid, nprocs = ClimaComms.init(context) assigns different mypid to the various processes and returns nprocs so that we can use this information if needed. This function is also responsible for distributing GPUs across processes, if relevant.","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"In this example, we used mypid to set up my_array in such a way that it would be different on different processes. We set up the array with ArrayType, obtained with array_type. This function provides the type to allocate the array on the correct device (CPU or GPU). Then, we applied reduce to sum them all. reduce collects the result to the root process, the one with pid = 1. For single-process runs, the only process is also a root process.","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"The code above works independently on the number of processes and over all the devices supported by ClimaComms.","category":"page"},{"location":"internals/#Backends-and-extensions","page":"Developing with ClimaComms","title":"Backends and extensions","text":"","category":"section"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"Except the most basic ones, ClimaComms computing devices and contexts are implemented as independent backends. For instance, ClimaComms provides an AbstractDevice interface for which CUDADevice is an implementation that depends on CUDA.jl. Scripts that use ClimaComms have to load the packages that power the desired backend (e.g., CUDA.jl has to be explicitly loaded if one wants to use CUDADevices).","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"When using the default context and device (as read from environment variables), ClimaComms can automatically load the required packages. To instruct ClimaComms to do, add ClimaComms.@import_required_backends at the beginning of your scripts. ClimaComms can also identify some cases when a package is required but not loaded and warn you about it.","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"Using non-trivial backends might require you to install CUDA.jl and/or MPI.jl in your environment.","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"Note: When using context to select the context, it is safe to always add ClimaComms.@import_required_backends at the top of your scripts. Do not add ClimaComms.@import_required_backends to library code (i.e., in src) because the macro requires dependencies that should not be satisfied in that instance.","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"Technically, ClimaComms backends are implemented in Julia extensions. There are two main reasons for this:","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"optional packages (such as CUDA and MPI) should not be hard dependencies of ClimaComms;\nimplementing code in extensions reduces the loading time for ClimaComms and downstream packages because it skips compilation of code that is not needed.","category":"page"},{"location":"internals/","page":"Developing with ClimaComms","title":"Developing with ClimaComms","text":"If you are implementing features for ClimaComms, make sure that your backend-specific code is in a Julia extension (in the ext folder).","category":"page"}]
}
